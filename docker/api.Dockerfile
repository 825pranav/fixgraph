# FixGraph API image, built and smoke-tested in CI (spec §5.9). Runs in fake mode by default:
# no GPU, no data, no API keys needed.
#
# CPU-only torch: uv.lock pins the CUDA build (cu130) for development. Here the lock is exported
# to a requirements file, torch/CUDA/triton lines are dropped, torch is installed from PyTorch's
# CPU index at the same version, then the rest of the locked dependencies are installed.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv

WORKDIR /app
ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    LLM__BACKEND=fake \
    UV_SYSTEM_PYTHON=1

COPY pyproject.toml uv.lock ./
RUN TORCH_VERSION="$(grep -A1 '^name = "torch"$' uv.lock | sed -n 's/^version = "\([0-9.]*\).*"/\1/p' | head -1)" \
 && uv export --locked --no-dev --no-hashes --no-annotate --no-header --no-emit-project \
      --format requirements-txt \
    | grep -v -E '^(torch==|nvidia-|triton==|cuda-)' > /tmp/requirements.txt \
 && uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && uv pip install -r /tmp/requirements.txt \
 && rm -rf /root/.cache

COPY src ./src
COPY configs ./configs

EXPOSE 8000
CMD ["uvicorn", "fixgraph.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
