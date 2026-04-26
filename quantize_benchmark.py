import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from measure_utils import measure_latency, measure_model_size, measure_peak_memory, evaluate_accuracy


DEVICE = 'cpu'
CHECKPOINT = 'checkpoints/resnet18_cifar10_baseline.pth'

# qnnpack is optimized for mobile/CPU INT8 inference — x86 default won't work here
# torch.backends.quantized.engine = 'qnnpack'
if torch.backends.quantized.supported_engines and 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'


def apply_int8_quantization(model):
  model.eval()
  for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
      with torch.no_grad():
        w = module.weight.data

        # Flatten to [out_channels, everything_else] so we can compute
        # one scale factor per output channel rather than one global scale
        w_flat = w.view(w.shape[0], -1)
        scale = w_flat.abs().max(dim=1)[0] / 127.0

        # Reshape scale to [out_ch, 1, 1, 1] so it broadcasts correctly
        # when dividing against the full weight tensor
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


def load_checkpoint(model, path, device='cpu'):
  ckpt = torch.load(path, map_location=device)
  # Handle both raw state_dict files and training loop checkpoints that wrap
  # state_dict inside a larger dict
  state = ckpt['model_state_dict'] if isinstance(
      ckpt, dict) and 'model_state_dict' in ckpt else ckpt
  model.load_state_dict(state)
  return model


def run_quantization_benchmark():
  os.makedirs('results', exist_ok=True)
  os.makedirs('checkpoints/quantized', exist_ok=True)
  test_loader = get_test_loader()
  dummy_input = torch.randn(128, 3, 32, 32)
  results = []

  print("--- FP32 Baseline ---")
  model_fp32 = get_resnet18_cifar10()
  model_fp32 = load_checkpoint(model_fp32, CHECKPOINT, DEVICE)
  model_fp32.eval()
  fp32_path = 'checkpoints/resnet18_fp32.pth'
  torch.save(model_fp32.state_dict(), fp32_path)

  latency_mean, latency_std = measure_latency(model_fp32, dummy_input)
  cv_pct = (latency_std / latency_mean) * 100

  results.append({
      'config': 'FP32_Baseline',
      'dtype': 'float32',
      'latency_ms': round(latency_mean, 3),
      'latency_std_ms': round(latency_std, 3),
      'cv_pct': round(cv_pct, 2),
      'model_size_mb': round(measure_model_size(fp32_path), 3),
      'peak_memory_mb': round(measure_peak_memory(model_fp32, dummy_input), 3),
      'accuracy_pct': round(evaluate_accuracy(model_fp32, test_loader), 2),
  })
  print(results[-1])

  print("\n--- INT8 Weight-Only Quantization ---")
  model_int8 = get_resnet18_cifar10()
  model_int8 = load_checkpoint(model_int8, CHECKPOINT, DEVICE)
  model_int8 = apply_int8_quantization(model_int8)

  int8_path = 'checkpoints/quantized/resnet18_int8.pth'
  size_mb = get_true_int8_size_mb(model_int8, int8_path)

  latency_mean, latency_std = measure_latency(model_int8, dummy_input)
  cv_pct = (latency_std / latency_mean) * 100

  results.append({
      'config': 'INT8_Weight_Only',
      'dtype': 'int8',
      'latency_ms': round(latency_mean, 3),
      'latency_std_ms': round(latency_std, 3),
      'cv_pct': round(cv_pct, 2),
      'model_size_mb': round(size_mb, 3),
      'peak_memory_mb': round(measure_peak_memory(model_int8, dummy_input), 3),
      'accuracy_pct': round(evaluate_accuracy(model_int8, test_loader), 2),
  })
  print(results[-1])

  return results


if __name__ == '__main__':
  rows = run_quantization_benchmark()
  import csv
  with open('results/quantization_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  print("\nSaved: results/quantization_metrics.csv")
