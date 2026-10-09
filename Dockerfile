FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY casework ./casework
RUN pip install --no-cache-dir . gunicorn==23.0.0
COPY artifacts ./artifacts
ENV MODEL_API_KEY=""
EXPOSE 5051
CMD ["gunicorn", "--bind", "0.0.0.0:5051", "--workers", "1", "--threads", "2", "casework.serve:create_app()"]
