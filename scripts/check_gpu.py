"""M0 machine check: CUDA-enabled torch must see the GPU (spec §5.3).

Standalone script run once on a new machine (`python scripts/check_gpu.py`, or the `gpu` poe
task) before any GPU stage. Uses only torch; imports nothing from fixgraph.
"""

# Imports: only sys and torch; this check does not depend on the fixgraph package.
import sys

import torch


# Print torch/CUDA info and return 0 if the GPU is usable, 1 if not (used as the exit code).
def main() -> int:
    print(f"torch version:   {torch.__version__}")
    print(f"built for CUDA:  {torch.version.cuda}")
    # Fail early with a fix hint if torch cannot see CUDA (usually the CPU-only wheel is installed).
    available = torch.cuda.is_available()
    print(f"CUDA available:  {available}")
    if not available:
        print(
            "ERROR: CUDA not available. The CPU wheel is probably installed; check the "
            "pytorch-cuda index in pyproject.toml, then `uv sync --reinstall-package torch`.",
            file=sys.stderr,
        )
        return 1

    # Print the GPU name, VRAM and compute capability so we know which models will fit.
    props = torch.cuda.get_device_properties(0)
    print(f"device:          {props.name}")
    print(f"total VRAM:      {props.total_memory / 1024**3:.2f} GiB")
    print(f"compute cap.:    {props.major}.{props.minor}")

    # Tiny kernel to prove the device actually executes work, not just enumerates.
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    print(f"matmul on GPU:   ok ({y:.1f})")
    return 0


# Exit with main()'s return code so scripts and CI can tell pass from fail.
if __name__ == "__main__":
    sys.exit(main())
