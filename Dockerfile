FROM python:3.13-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN python -m pip install '.[dev]'
CMD ["python", "-m", "failure_lab.runner", "run", "all"]
