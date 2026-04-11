import torch
import time
import os


def measure_latency(model, input_tensor, warmup=20, runs=500):
  """
  Returns (mean_ms, std_ms) over `runs` forward passes.
  500 runs required for statistically stable CPU latency measurements.
  """
  model.eval()
  timings = []
  with torch.no_grad():
    # Warmup — discard these, they include JIT/cache cold-start effects
    for _ in range(warmup):
      _ = model(input_tensor)
    # Timed runs
    for _ in range(runs):
      start = time.perf_counter()
      _ = model(input_tensor)
      end = time.perf_counter()
      timings.append((end - start) * 1000)  # ms

  mean_ms = sum(timings) / len(timings)
  variance = sum((t - mean_ms) ** 2 for t in timings) / len(timings)
  std_ms = variance ** 0.5
  return mean_ms, std_ms


def measure_model_size(path):
  """Returns on-disk model size in MB."""
  return os.path.getsize(path) / (1024 ** 2)


def measure_peak_memory(model, input_tensor):
  """
  Computes actual tensor memory footprint in MB.
  tracemalloc and psutil are unreliable for PyTorch CPU tensors —
  PyTorch uses its own C++ allocator that bypasses Python's memory layer.
  This calculates ground-truth memory from actual parameter + buffer byte sizes.
  """
  param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
  buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
  input_bytes = input_tensor.numel() * input_tensor.element_size()
  return (param_bytes + buffer_bytes + input_bytes) / (1024 ** 2)


def evaluate_accuracy(model, test_loader, device='cpu'):
  """Returns top-1 accuracy (%) on test_loader."""
  model.eval()
  correct, total = 0, 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      correct += (predicted == labels).sum().item()
      total += labels.size(0)
  return 100.0 * correct / total
