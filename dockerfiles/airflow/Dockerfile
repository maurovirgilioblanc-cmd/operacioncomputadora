FROM apache/airflow:3.0.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    imbalanced-learn==0.12.3 \
    xgboost==2.0.3 \
    catboost==1.2.5 \
    boto3 \
    mlflow==2.19.0 \
    apache-airflow-providers-amazon