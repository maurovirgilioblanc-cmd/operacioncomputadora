from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MINIO_ENDPOINT = "http://s3:9000"
DATA_BUCKET = "data"
DATA_KEY = "stroke-data.csv"
RANDOM_STATE = 42

def upload_dataset():
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    # Verifica si ya existe
    try:
        s3.head_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
        print("Dataset ya existe en MinIO, se omite la subida.")
    except Exception:
        raise FileNotFoundError(
            f"El archivo {DATA_KEY} no está en el bucket '{DATA_BUCKET}'. "
            "Subilo manualmente a MinIO primero."
        )

def train_and_register():
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import boto3
    from io import StringIO

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    # --- Cargar datos desde MinIO ---
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    # --- Feature engineering (igual al notebook) ---
    df = df.drop(columns=["id"])
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 120],
                              labels=["Joven", "Adulto", "Mayor", "Anciano"])
    df["age_group"] = df["age_group"].astype(str)
    df["avg_glucose_level"] = np.clip(df["avg_glucose_level"], 50, 300)
    df["has_risk_factors"] = np.where(
        (df["hypertension"] == 1) | (df["heart_disease"] == 1), 1, 0
    )
    df["is_female"] = np.where(df["gender"] == "Female", 1, 0)
    df = df.drop(columns=["gender"])
    df["work_type"] = df["work_type"].replace({"Never_worked": "children"})

    # --- Split ---
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --- Imputación BMI con mediana del train ---
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    # --- Preprocessor para LR (igual al notebook) ---
    cols_eliminar_lr = ["heart_disease", "bmi", "smoking_status", "age_group"]
    num_features_lr = (
        X.select_dtypes(exclude="object")
        .columns.drop(cols_eliminar_lr, errors="ignore")
        .tolist()
    )

    preprocessor_lr = ColumnTransformer([
        ("num", StandardScaler(), num_features_lr),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), []),
    ])

    # --- Pipeline con SMOTE ---
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor_lr),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(
            C=0.1,
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # --- Entrenar ---
    pipeline.fit(X_train, y_train)

    # --- Threshold óptimo ---
    from sklearn.metrics import precision_recall_curve
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    thr_candidates = thresholds[recall[:-1] >= 0.90]
    best_threshold = float(thr_candidates.max()) if len(thr_candidates) > 0 else 0.5

    y_pred = (y_proba >= best_threshold).astype(int)

    # --- Métricas ---
    auc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # --- Registrar en MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stroke-prediction")

    with mlflow.start_run(run_name="logistic_regression_smote"):
        mlflow.log_param("C", 0.1)
        mlflow.log_param("penalty", "l1")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_param("bmi_median", bmi_median)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="stroke-predictor",
        )

        print(f"AUC: {auc:.4f} | Recall: {rec:.4f} | Threshold: {best_threshold:.5f}")

with DAG(
    dag_id="train_stroke_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["stroke", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="check_dataset", python_callable=upload_dataset)
    t2 = PythonOperator(task_id="train_and_register", python_callable=train_and_register)
    t1 >> t2