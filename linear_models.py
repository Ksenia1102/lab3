import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _standardize_params(X):
    """Среднее и стандартное отклонение по train (нулевое отклонение заменяем на 1)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _unscale(weights_scaled, bias_scaled, mean, std):
    """Переводит веса из стандартизованных признаков обратно в исходные единицы."""
    weights = weights_scaled / std
    bias = float(bias_scaled - np.sum(weights_scaled * mean / std))
    return weights, bias


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_linear_gd(X, y, learning_rate, epochs):
    """Пакетный градиентный спуск по MSE. Возвращает веса, смещение и историю MSE."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(epochs):
        error = X @ weights + bias - y
        loss_history.append(float(np.mean(error ** 2)))

        # градиент MSE: d/dw = 2/n * X^T (y_hat - y), d/db = 2/n * sum(y_hat - y)
        weights -= learning_rate * 2 / n_samples * (X.T @ error)
        bias -= learning_rate * 2 / n_samples * error.sum()

    return weights, float(bias), loss_history


def fit_logistic_gd(X, y, learning_rate, epochs):
    """Пакетный градиентный спуск по бинарной кросс-энтропии."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    eps = 1e-12  # чтобы не получить log(0)

    for _ in range(epochs):
        proba = _sigmoid(X @ weights + bias)
        clipped = np.clip(proba, eps, 1 - eps)
        loss_history.append(float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))))

        # градиент кросс-энтропии: X^T (p - y) / n
        error = proba - y
        weights -= learning_rate * (X.T @ error) / n_samples
        bias -= learning_rate * error.mean()

    return weights, float(bias), loss_history


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test.

    Признаки стандартизуются по train внутри функции (если они уже масштабированы,
    это ничего не меняет), а веса возвращаются в исходных единицах признаков.
    """
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    mean, std = _standardize_params(X)
    weights_s, bias_s, loss_history = fit_linear_gd(
        (X - mean) / std, y, data.learning_rate, data.epochs
    )
    weights, bias = _unscale(weights_s, bias_s, mean, std)

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=X_test @ weights + bias,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test.

    Масштабирование и возврат весов в исходных единицах - как в линейной регрессии.
    Порог классификации 0.5.
    """
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    mean, std = _standardize_params(X)
    weights_s, bias_s, loss_history = fit_logistic_gd(
        (X - mean) / std, y, data.learning_rate, data.epochs
    )
    weights, bias = _unscale(weights_s, bias_s, mean, std)

    probabilities = _sigmoid(X_test @ weights + bias)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=(probabilities >= 0.5).astype(int),
    )


def classification_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    """ROC-AUC считается по вероятностям, остальные метрики - по бинарным предсказаниям."""
    y_true = np.asarray(y_true).ravel()
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions)),
        recall=float(recall_score(y_true, predictions)),
        f1=float(f1_score(y_true, predictions)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    # масштаб важен для логистической регрессии и KNN, дереву он безразличен
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        metrics[name] = classification_metrics(y_test, probabilities, predictions)

    return BaselineComparisonResult(**metrics)
