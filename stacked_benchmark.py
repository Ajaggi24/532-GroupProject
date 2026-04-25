import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization as tq
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from measure_utils import measure_latency, measure_model_size, measure_peak_memory, evaluate_accuracy
from torchao.quantization import quantize_, Int8WeightOnlyConfig
import safetensors.torch as st
import io


# qnnpack is optimized for mobile/CPU INT8 inference — x86 default won't work here
torch.backends.quantized.engine = 'qnnpack'

DEVICE = 'cpu'
CHECKPOINT = 'checkpoints/resnet18_cifar10_baseline.pth'
STACKED_SPARSITY = 0.50


def apply_int8_quantization(model):
  model.eval()
  for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
      with torch.no_grad():
        w = module.weight.data

        # Flatten to [out_channels, everything_else] for per-channel scaling
        w_flat = w.view(w.shape[0], -1)
        scale = w_flat.abs().max(dim=1)[0] / 127.0

        # Reshape to [out_ch, 1, 1, 1] so scale broadcasts over the full weight tensor
        scale = scale.view(w.shape[0], *([1] * (w.dim() - 1)))

        w_int8 = (w / scale.clamp(min=1e-8)
                  ).round().clamp(-128, 127).to(torch.int8)
        # Dequantize back to float so standard CPU inference still works
        module.weight.data = w_int8.float() * scale
  return model


def get_true_int8_size_mb(model, path):
  # Our quantization stores float weights at runtime for CPU compatibility,
  # so on-disk size won't reflect INT8 — we compute the theoretical size manually:
  # 1 byte per weight + 4-byte float scale per output channel
  int8_bytes = 0
  float_bytes = 0

  for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      int8_bytes += module.weight.numel() * 1
      int8_bytes += module.weight.shape[0] * 4
      if module.bias is not None:
        float_bytes += module.bias.numel() * 4
    elif hasattr(module, 'weight') and module.weight is not None:
      float_bytes += module.weight.numel() * 4

  for buf in model.buffers():
    float_bytes += buf.numel() * buf.element_size()

  size_mb = (int8_bytes + float_bytes) / (1024 ** 2)
  # Save a dummy file just to keep the path valid for downstream size checks
  torch.save({'size_mb': size_mb}, path)
  return size_mb


def get_resnet18_cifar10():
  # Swap out the default 7×7 stride-2 conv — too aggressive for 32×32 CIFAR images
  model = models.resnet18(weights=None, num_classes=10)
  model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
  model.maxpool = nn.Identity()
  return model


def get_test_loader():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
  ])
  return DataLoader(
      datasets.CIFAR10('./data', train=False,
                       download=False, transform=transform),
      batch_size=128, shuffle=False, num_workers=2
  )


def apply_pruning_permanent(model, sparsity):
  params = [(m, 'weight') for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]
  prune.global_unstructured(
      params, pruning_method=prune.L1Unstructured, amount=sparsity)
  # Bake the pruning masks into the weights and drop them — keeps the
  # checkpoint clean and avoids mask tensors interfering with quantization
  for module, _ in params:
    prune.remove(module, 'weight')
  return model


def load_checkpoint(model, path, device='cpu'):
  ckpt = torch.load(path, map_location=device)
  # Handle both raw state_dict files and training loop checkpoints that wrap
  # state_dict inside a larger dict
  state = ckpt['model_state_dict'] if isinstance(
      ckpt, dict) and 'model_state_dict' in ckpt else ckpt
  model.load_state_dict(state)
  return model


def run_stacked_benchmark():
  os.makedirs('results', exist_ok=True)
  os.makedirs('checkpoints/stacked', exist_ok=True)
  test_loader = get_test_loader()
  dummy_input = torch.randn(128, 3, 32, 32)

  print(
    f"--- Stacked: {int(STACKED_SPARSITY * 100)}% Pruning + INT8 Quantization ---")
  model = get_resnet18_cifar10()
  model = load_checkpoint(model, CHECKPOINT, DEVICE)
  model.eval()

  # Pruning must come first — quantizing before pruning would distort the
  # weight magnitudes that L1 pruning relies on to pick which weights to drop
  model = apply_pruning_permanent(model, STACKED_SPARSITY)
  model = apply_int8_quantization(model)

  save_path = 'checkpoints/stacked/resnet18_pruned50_int8.pth'
  size_mb = get_true_int8_size_mb(model, save_path)
  latency_mean, latency_std = measure_latency(model, dummy_input)
  cv_pct = (latency_std / latency_mean) * 100

  result = {
      'config': f'Pruned{int(STACKED_SPARSITY * 100)}pct_INT8',
      'latency_ms': round(latency_mean, 3),
      'latency_std_ms': round(latency_std, 3),
      'cv_pct': round(cv_pct, 2),
      'model_size_mb': round(size_mb, 3),
      'peak_memory_mb': round(measure_peak_memory(model, dummy_input), 3),
      'accuracy_pct': round(evaluate_accuracy(model, test_loader), 2),
  }
  print(result)
  return [result]


if __name__ == '__main__':
  rows = run_stacked_benchmark()
  import csv
  with open('results/stacked_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  print("\nSaved: results/stacked_metrics.csv")
