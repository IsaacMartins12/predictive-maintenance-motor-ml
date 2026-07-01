"""
Script de treinamento multi-modelo com MLflow tracking.
Treina todos os modelos habilitados em params.yaml e loga cada um como um run separado no MLflow.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ---------- Configurações ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "industrial_motor_sensor_data_8000.csv"
MODEL_DIR = BASE_DIR / "models"
PARAMS_PATH = BASE_DIR / "params.yaml"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive-maintenance-motor")

# Carrega configuração
with open(PARAMS_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_PARAMS = config["data"]
MODELS_CONFIG = config["models"]


def build_model(name: str, params: dict):
    """Instancia o modelo a partir do nome e parâmetros do params.yaml."""
    # Remove 'enabled' dos params antes de passar pro construtor
    model_params = {k: v for k, v in params.items() if k != "enabled"}

    if name == "logistic_regression":
        return LogisticRegression(**model_params)
    elif name == "decision_tree":
        return DecisionTreeClassifier(**model_params)
    elif name == "random_forest":
        return RandomForestClassifier(**model_params)
    elif name == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", **model_params)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")


def train_and_log(model_name: str, model_params: dict, X_train, X_test, y_train, y_test, features):
    """Treina um modelo e loga tudo no MLflow como um run."""
    print(f"\n{'='*60}")
    print(f"  Treinando: {model_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):
        # Tags
        mlflow.set_tag("model_type", model_name)

        # Log de parâmetros
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", DATA_PARAMS["test_size"])
        mlflow.log_param("random_state", DATA_PARAMS["random_state"])
        mlflow.log_param("dataset_size", len(X_train) + len(X_test))
        mlflow.log_param("features", list(features))

        for k, v in model_params.items():
            if k != "enabled":
                mlflow.log_param(k, v)

        # Instancia e treina
        model = build_model(model_name, model_params)

        # XGBoost precisa de labels encodados
        le = None
        if model_name == "xgboost":
            le = LabelEncoder()
            y_train_fit = le.fit_transform(y_train)
            model.fit(X_train, y_train_fit)
            pred = le.inverse_transform(model.predict(X_test))
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted")
        rec = recall_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_weighted", prec)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.log_metric("f1_weighted", f1)

        print(f"\n  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"\n{classification_report(y_test, pred)}")

        # Feature importance (se disponível)
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(features, [float(x) for x in model.feature_importances_]))
            mlflow.log_dict(importance, "feature_importance.json")
            print(f"  Feature Importance: {importance}")

        # Log do modelo no MLflow
        mlflow.sklearn.log_model(model, "model")

        # Salva modelo localmente
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"  Modelo salvo em: {model_path}")

    return {"model_name": model_name, "accuracy": acc, "f1_weighted": f1}


def main():
    # ---------- MLflow setup ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ---------- Carregamento dos dados ----------
    print(f"Carregando dados de: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Distribuição das classes:\n{df['Label'].value_counts()}\n")

    # ---------- Preparação ----------
    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_PARAMS["test_size"],
        random_state=DATA_PARAMS["random_state"],
        stratify=y,
    )

    # ---------- Treina cada modelo habilitado ----------
    results = []
    for model_name, model_params in MODELS_CONFIG.items():
        if not model_params.get("enabled", True):
            print(f"\n[SKIP] {model_name} está desabilitado no params.yaml")
            continue

        result = train_and_log(
            model_name, model_params,
            X_train, X_test, y_train, y_test,
            features=X.columns,
        )
        results.append(result)

    # ---------- Resumo ----------
    print(f"\n{'='*60}")
    print("  RESUMO DOS EXPERIMENTOS")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    print(results_df.to_string(index=False))
    print(f"\n  Melhor modelo: {results_df.iloc[0]['model_name']} (F1: {results_df.iloc[0]['f1_weighted']:.4f})")
    print(f"\n  Visualize todos os runs em: {MLFLOW_TRACKING_URI}")
    print("\nTreinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()
