import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics,
    compare_classical_models,
    fit_linear_gd,
    train_linear_regression,
    train_logistic_classifier,
)

BASE_DIR = Path(__file__).parent
TEST_SIZE = 0.2
RANDOM_STATE = 42
VAL_SIZE = 0.2  # доля validation внутри train для экспериментов с параметрами


def prepare_regression():
    df = pd.read_csv(BASE_DIR / "Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    X = df.drop(columns="Performance Index")
    y = df["Performance Index"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


NUMERIC_COLUMNS = [
    "Duration_of_Credit_monthly", "Credit_Amount", "Instalment_per_cent",
    "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank", "No_of_dependents",
]


def prepare_classification():
    df = pd.read_csv(BASE_DIR / "german.csv", sep=";")
    X = df.drop(columns="Creditability")
    y = df["Creditability"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def encode(X_train, X_test):
    """Числовые признаки масштабируются, остальные (коды категорий) кодируются one-hot.
    Параметры преобразования берутся только из train."""
    categorical = [c for c in X_train.columns if c not in NUMERIC_COLUMNS]
    transformer = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return transformer.fit_transform(X_train), transformer.transform(X_test)


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def run_regression():
    X_train, X_test, y_train, y_test = prepare_regression()
    # масштабирование делает сама функция (по среднему и std train), сюда идут сырые признаки
    X_train_np, X_test_np = X_train.to_numpy(float), X_test.to_numpy(float)

    def fit(features_train, features_test, **params):
        return train_linear_regression(
            RegressionInput(features_train, y_train.to_numpy(), features_test, y_test.to_numpy(), **params)
        )

    result = fit(X_train_np, X_test_np)
    test_mse = mse(y_test, result.predictions)
    history = result.loss_history

    print("=== Линейная регрессия ===")
    print(f"train/test: {len(X_train)}/{len(X_test)}")
    print("признаки:", list(X_train.columns))
    print("weights (исходные единицы):", np.round(result.weights, 4).tolist(), "bias:", round(result.bias, 4))
    print("weights (на 1 std признака):", np.round(result.weights * X_train_np.std(axis=0), 4).tolist())
    print("std признаков:", np.round(X_train_np.std(axis=0), 4).tolist(), "| mean(y_train):", round(float(y_train.mean()), 4))
    for epoch in (1, 10, 50, 100, 400, 800):
        print(f"train MSE, эпоха {epoch}: {history[epoch - 1]:.4f}")
    print(f"train MSE (последняя эпоха): {history[-1]:.4f}")
    print(f"test MSE: {test_mse:.4f}")

    # проверка: аналитическое решение sklearn (не используется внутри своей модели)
    reference = LinearRegression().fit(X_train_np, y_train)
    print(f"test MSE (sklearn LinearRegression, для сверки): {mse(y_test, reference.predict(X_test_np)):.4f}")

    # эксперимент 1: спуск на сырых признаках, без масштабирования (ядро fit_linear_gd, в обход train_*)
    with np.errstate(all="ignore"):
        *_, raw_history = fit_linear_gd(X_train_np, y_train.to_numpy(float), 0.05, 800)
    print("без масштабирования: train MSE на эпохах 1/2/3:", [float(f"{v:.4g}") for v in raw_history[:3]],
          "| последняя:", float(f"{raw_history[-1]:.4g}"))

    # эксперимент 2: другие learning rate / число эпох; оцениваем на validation из train, test не трогаем
    X_in, X_val, y_in, y_val = train_test_split(X_train_np, y_train.to_numpy(), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    for lr, epochs in ((0.01, 800), (0.05, 800), (0.1, 800), (0.05, 200), (0.05, 3000)):
        r = train_linear_regression(RegressionInput(X_in, y_in, X_val, y_val, lr, epochs))
        print(f"lr={lr}, epochs={epochs}: validation MSE = {mse(y_val, r.predictions):.4f}")

    return test_mse


def run_classification():
    X_train, X_test, y_train, y_test = prepare_classification()
    X_train_np, X_test_np = encode(X_train, X_test)
    print(f"признаков после one-hot: {X_train_np.shape[1]}")

    def fit(**params):
        return train_logistic_classifier(
            ClassificationInput(X_train_np, y_train.to_numpy(), X_test_np, y_test.to_numpy(), **params)
        )

    result = fit()
    own = classification_metrics(y_test, result.probabilities, result.predictions)
    history = result.loss_history

    print("\n=== Логистическая регрессия ===")
    print(f"train/test: {len(X_train)}/{len(X_test)}, доля класса 1: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
    for epoch in (1, 10, 100, 500, 1000):
        print(f"train log-loss, эпоха {epoch}: {history[epoch - 1]:.4f}")
    print("own:", {k: round(v, 4) for k, v in asdict(own).items()})
    print(f"доля единиц в предсказаниях: {result.predictions.mean():.3f}; "
          f"accuracy при предсказании всегда 1: {y_test.mean():.3f}")

    # эксперимент: другие learning rate / число эпох; оцениваем на validation из train, test не трогаем
    X_in, X_val, y_in, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    X_in_np, X_val_np = encode(X_in, X_val)
    print(f"внутренний train/validation: {len(X_in)}/{len(X_val)}")
    for lr, epochs in ((0.01, 1000), (0.1, 200), (0.1, 1000), (0.5, 1000), (0.1, 5000)):
        r = train_logistic_classifier(
            ClassificationInput(X_in_np, y_in.to_numpy(), X_val_np, y_val.to_numpy(), lr, epochs)
        )
        m = classification_metrics(y_val, r.probabilities, r.predictions)
        print(f"lr={lr}, epochs={epochs}: train loss={r.loss_history[-1]:.4f}, val auc={m.roc_auc:.4f}, "
              f"val acc={m.accuracy:.4f}, val f1={m.f1:.4f}")

    baseline = compare_classical_models(
        ClassificationInput(X_train_np, y_train.to_numpy(), X_test_np, y_test.to_numpy())
    )
    print("\n=== Сравнение моделей (test) ===")
    print(f"{'модель':<22}{'ROC-AUC':>9}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    rows = {"own_logistic": own, **asdict(baseline)}
    for name, m in rows.items():
        m = asdict(m) if not isinstance(m, dict) else m
        print(f"{name:<22}{m['roc_auc']:>9.4f}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}")

    return {name: (asdict(m) if not isinstance(m, dict) else m) for name, m in rows.items()}


if __name__ == "__main__":
    results = {
        "linear_regression": {"test_mse": run_regression()},
        "classification": run_classification(),
    }
    with open(BASE_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nexperiment_results.json записан")