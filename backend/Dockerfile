FROM python:3.11-slim

WORKDIR /app

# Install system deps needed to build some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository backend files (app/, model.py, train scripts, content, etc.)
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# For convenience in dev images, run a quick training to create a model artifact so
# the service can serve predictions immediately. This uses the --quick flag to keep
# build time low. In production you may remove or replace this step with a model
# deployment artifact.
RUN python train_and_save.py --quick --out app/model/model.joblib || echo "Training at build time failed; continuing without model"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
