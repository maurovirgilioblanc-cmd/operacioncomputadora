FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    mlflow==2.19.0 \
    boto3 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4  \
    imbalanced-learn==0.12.3

COPY app.py .

EXPOSE 8800