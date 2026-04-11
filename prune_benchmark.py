import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from measure_utils import measure_latency, measure_model_size, measure_peak_memory, evaluate_accuracy

DEVICE = 'cpu'
CHECKPOINT = 'checkpoints/resnet18_cifar10_baseline.pth'
SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90]


def get_resnet18_cifar10():
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
  test_set = datasets.CIFAR10(
    './data', train=False, download=False, transform=transform)
  return DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)


def get_prunable_parameters(model):
  """Collect all (module, 'weight') pairs from Conv2d and Linear layers."""
  params_to_prune = []
  for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      params_to_prune.append((module, 'weight'))
  return params_to_prune


def apply_global_l1_pruning(model, sparsity):
  """Apply global unstructured L1 pruning then make it permanent."""
  if sparsity == 0.0:
    return model
  params = get_prunable_parameters(model)
  prune.global_unstructured(
      params,
      pruning_method=prune.L1Unstructured,
      amount=sparsity,
  )
  # Make pruning permanent — remove masks, zero weights are baked in
  for module, _ in params:
    prune.remove(module, 'weight')
  return model


def compute_actual_sparsity(model):
  """Returns the fraction of zero weights across all prunable layers."""
  total, zeros = 0, 0
  for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      total += module.weight.numel()
      zeros += (module.weight == 0).sum().item()
  return zeros / total if total > 0 else 0.0


def load_checkpoint(model, path, device='cpu'):
  ckpt = torch.load(path, map_location=device)
  state = ckpt['model_state_dict'] if isinstance(
    ckpt, dict) and 'model_state_dict' in ckpt else ckpt
  model.load_state_dict(state)
  return model


def run_pruning_benchmark():
  os.makedirs('results', exist_ok=True)
  os.makedirs('checkpoints/pruned', exist_ok=True)
  test_loader = get_test_loader()
  dummy_input = torch.randn(128, 3, 32, 32)
  results = []

  for sparsity in SPARSITY_LEVELS:
    print(f"\n--- Pruning sparsity: {int(sparsity * 100)}% ---")
    model = get_resnet18_cifar10()
    model = load_checkpoint(model, CHECKPOINT, DEVICE)
    model = apply_global_l1_pruning(model, sparsity)
    model.eval()

    # Save pruned model
    save_path = f'checkpoints/pruned/resnet18_pruned_{int(sparsity * 100)}.pth'
    torch.save(model.state_dict(), save_path)

    actual_sparsity = compute_actual_sparsity(model)
    latency_mean, latency_std = measure_latency(model, dummy_input)
    cv_pct = (latency_std / latency_mean) * 100   # Coefficient of Variation
    disk_size = measure_model_size(save_path)
    peak_mem = measure_peak_memory(model, dummy_input)
    accuracy = evaluate_accuracy(model, test_loader)

    row = {
        'config': f'Pruned_{int(sparsity * 100)}pct',
        'target_sparsity': sparsity,
        'actual_sparsity': round(actual_sparsity, 4),
        'latency_ms': round(latency_mean, 3),
        'latency_std_ms': round(latency_std, 3),
        'cv_pct': round(cv_pct, 2),  # should be < 5% for stable results
        'model_size_mb': round(disk_size, 3),
        'peak_memory_mb': round(peak_mem, 3),
        'accuracy_pct': round(accuracy, 2),
    }
    results.append(row)
    print(row)

  return results


if __name__ == '__main__':
  rows = run_pruning_benchmark()
  import csv
  with open('results/pruning_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  print("\nSaved: results/pruning_metrics.csv")
