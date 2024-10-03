import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings

warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла.

    :param filepath: Путь к файлу.
    :return: DataFrame с загруженными данными.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Файл '{filepath}' успешно загружен.")
    except Exception as e:
        raise FileNotFoundError(f"Не удалось загрузить файл: {e}")
    return data


def check_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка и обработка пропущенных значений.

    :param data: Исходные данные.
    :return: DataFrame без пропущенных значений.
    """
    missing = data.isnull().sum()
    print("Пропущенные значения по столбцам:")
    print(missing)
    if missing.any():
        data = data.dropna()
        print("Строки с пропущенными значениями удалены.")
    else:
        print("Пропущенные значения отсутствуют.")
    return data


def check_multicollinearity(data: pd.DataFrame) -> None:
    """
    Проверка мультиколлинеарности между признаками.

    :param data: Исходные данные.
    """
    features = data.drop(["id", "cardio"], axis=1)
    corr_matrix = features.corr()
    print("Матрица корреляций:")
    print(corr_matrix)

    rank = np.linalg.matrix_rank(corr_matrix)
    det = np.linalg.det(corr_matrix)
    print(f"Ранг матрицы корреляций: {rank}")
    print(f"Определитель матрицы корреляций: {det}")

    if det == 0:
        print("Матрица корреляций вырожденная, присутствует мультиколлинеарность.")
    else:
        print("Матрица корреляций невырожденная, чистой мультиколлинеарности нет.")

    # Визуализация матрицы корреляций
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Матрица корреляций признаков")
    plt.show()


def separate_features_target(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Разделение данных на признаки и целевую переменную.

    :param data: Исходные данные.
    :return: Кортеж из массивов признаков (X) и целевой переменной (y).
    """
    X = data.drop(["id", "cardio"], axis=1).values
    y = data["cardio"].values
    return X, y


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Стандартизация признаков без использования готовых функций.

    :param X: Массив признаков.
    :return: Стандартизированный массив признаков.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - mean) / std
    return X_std


def train_test_split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделение данных на обучающую и тестовую выборки.

    :param X: Массив признаков.
    :param y: Целевая переменная.
    :param test_size: Доля тестовой выборки.
    :param random_state: Сид для воспроизводимости.
    :return: Кортеж из X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_naive_bayes(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = "Gaussian"
) -> GaussianNB | MultinomialNB | BernoulliNB:
    """
    Обучение наивного байесовского классификатора.
:param X_train: Обучающие признаки.
    :param y_train: Обучающая целевая переменная.
    :param model_type: Тип наивного Байеса ('Gaussian', 'Multinomial', 'Bernoulli').
    :return: Обученная модель.
    """
    if model_type == "Gaussian":
        model = GaussianNB()
    elif model_type == "Multinomial":
        model = MultinomialNB()
    elif model_type == "Bernoulli":
        model = BernoulliNB()
    else:
        raise ValueError(
            "Неверный тип наивного Байеса. Выберите 'Gaussian', 'Multinomial' или 'Bernoulli'."
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> None:
    """
    Оценка качества модели и построение ROC-кривой с названием модели.

    :param model: Обученная модель.
    :param X_test: Тестовые признаки.
    :param y_test: Тестовая целевая переменная.
    :param model_name: Название модели для отображения на графике.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Точность модели {model_name}: {acc:.4f}")
    print(f"AUC-ROC модели {model_name}: {auc:.4f}")
    print("Отчет классификации:")
    print(classification_report(y_test, y_pred))

    # Визуализация ROC-кривой
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-кривая {model_name}")
    plt.legend(loc="best")
    plt.show()


def train_decision_tree(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> DecisionTreeClassifier:
    """
    Обучение решающего дерева.

    :param X_train: Обучающие признаки.
    :param y_train: Обучающая целевая переменная.
    :param params: Гиперпараметры модели.
    :return: Обученная модель решающего дерева.
    """
    if params:
        model = DecisionTreeClassifier(**params)
    else:
        model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def plot_feature_importances(
    model: DecisionTreeClassifier | RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 3,
) -> None:
    """
    Вывод важности признаков.

    :param model: Обученная модель с атрибутом feature_importances_.
    :param feature_names: Список имен признаков.
    :param top_n: Количество топовых признаков для отображения.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:top_n]]
    print(f"Топ-{top_n} важных признаков:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")

    # Визуализация
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[imp for _, imp in top_features], y=[feat for feat, _ in top_features])
    plt.title(f"Важность топ-{top_n} признаков")
    plt.xlabel("Важность")
    plt.ylabel("Признаки")
    plt.show()


def hyperparameter_tuning(
    model, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray
) -> DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier:
    """
    Подбор гиперпараметров с помощью GridSearchCV.

    :param model: Модель для настройки.
    :param param_grid: Сетка гиперпараметров.
    :param X_train: Обучающие признаки.
    :param y_train: Обучающая целевая переменная.
    :return: Модель с оптимальными гиперпараметрами.
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> RandomForestClassifier:
    """
    Обучение модели случайного леса.
:param X_train: Обучающие признаки.
    :param y_train: Обучающая целевая переменная.
    :param params: Гиперпараметры модели.
    :return: Обученная модель случайного леса.
    """
    if params:
        model = RandomForestClassifier(**params)
    else:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> GradientBoostingClassifier:
    """
    Обучение модели градиентного бустинга.

    :param X_train: Обучающие признаки.
    :param y_train: Обучающая целевая переменная.
    :param params: Гиперпараметры модели.
    :return: Обученная модель градиентного бустинга.
    """
    if params:
        model = GradientBoostingClassifier(**params)
    else:
        model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """
    Основная функция выполнения всех заданий.
    """
    # Путь к датасету
    filepath = "data/cvd.xls - cvd.xls.csv"

    # Задание 1: Загрузка данных и проверка пропущенных значений
    print("Задание 1: Загрузка данных и проверка пропущенных значений")
    data = load_data(filepath)
    data = check_missing_values(data)
    print("-" * 50)

    # Задание 2: Проверка мультиколлинеарности
    print("Задание 2: Проверка мультиколлинеарности")
    check_multicollinearity(data)
    print("-" * 50)

    # Задание 3: Стандартизация
    print("Задание 3: Стандартизация признаков")
    X, y = separate_features_target(data)
    X_std = standardize_features(X)
    print("Стандартизация завершена.")
    print("-" * 50)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split_data(X_std, y)

    # Задание 4: Наивный байесовский классификатор
    print("Задание 4: Наивный байесовский классификатор")
    # 4.1 Выбор типа наивного Байеса
    nb_model = train_naive_bayes(X_train, y_train, model_type="Gaussian")
    print("Наивный Байесовский классификатор обучен (GaussianNB).")
    # 4.2 Оценка качества модели
    evaluate_model(nb_model, X_test, y_test, model_name="Gaussian Naive Bayes")
    print("-" * 50)

    # Задание 5: Решающее дерево
    print("Задание 5: Решающее дерево")
    feature_names = data.drop(["id", "cardio"], axis=1).columns.tolist()
    # 5.1 Обучение дерева без гиперпараметров
    dt_model = train_decision_tree(X_train, y_train)
    print("Решащее дерево обучено без настройки гиперпараметров.")
    print(f"Параметры модели: {dt_model.get_params()}")
    print("Оценка качества модели:")
    evaluate_model(
        dt_model,
        X_test,
        y_test,
        model_name="Decision Tree without hyperparams tuning",
    )
    # Сравнение с наивным Байесом
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    print(f"Точность наивного Байеса: {nb_acc:.4f}")
    print(f"Точность решающего дерева: {dt_acc:.4f}")
    # 5.2 Подбор гиперпараметров с помощью GridSearchCV
    print("5.2 Подбор гиперпараметров для решающего дерева")
    param_grid_dt = {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    dt_best = hyperparameter_tuning(
        DecisionTreeClassifier(), param_grid_dt, X_train, y_train
    )
    print("Решащее дерево с оптимальными гиперпараметрами обучено.")
    print(f"Оценка качества оптимизированной модели:")
    evaluate_model(
        dt_best, X_test, y_test, model_name="Decision Tree with hyperparams tuning"
    )

    # 5.3 Важность признаков
    print("5.3 Важность признаков для оптимизированного решающего дерева")
    plot_feature_importances(dt_best, feature_names)
    print("-" * 50)
    # Задание 6: Случайный лес и градиентный бустинг
    print("Задание 6: Случайный лес и градиентный бустинг")
    # 6.1 Случайный лес
    print("6.1 Случайный лес без настройки гиперпараметров")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_name="Random Forest without hyperparams tuning")

    print("6.1 Подбор гиперпараметров для случайного леса")
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_best = hyperparameter_tuning(
        RandomForestClassifier(), param_grid_rf, X_train, y_train
    )
    print("Случайный лес с оптимальными гиперпараметрами обучен.")
    evaluate_model(rf_best, X_test, y_test, model_name="Random Forest with hyperparams tuning")
    print("Важность признаков для случайного леса:")
    plot_feature_importances(rf_best, feature_names)

    # Сравнение с решающим деревом
    rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
    dt_best_acc = accuracy_score(y_test, dt_best.predict(X_test))
    print(f"Точность оптимизированного случайного леса: {rf_acc:.4f}")
    print(f"Точность оптимизированного решающего дерева: {dt_best_acc:.4f}")

    # 6.2 Градиентный бустинг
    print("6.2 Градиентный бустинг")
    print("Обучение первой модели градиентного бустинга")
    gb_model1 = train_gradient_boosting(X_train, y_train)
    evaluate_model(
        gb_model1, X_test, y_test, model_name="Gradient Boosting without hyperparams tuning"
    )

    print("Обучение второй модели градиентного бустинга с настройкой гиперпараметров")
    param_grid_gb = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }
    gb_best = hyperparameter_tuning(
        GradientBoostingClassifier(), param_grid_gb, X_train, y_train
    )
    evaluate_model(
        gb_best, X_test, y_test, model_name="Gradient Boosting with hyperparams tuning"
    )

    # Сравнение сo случайным лесом
    gb_best_acc = accuracy_score(y_test, gb_best.predict(X_test))
    print(f"Точность оптимизированного градиентного бустинга: {gb_best_acc:.4f}")
    print(f"Точность оптимизированного случайного леса: {rf_acc:.4f}")


if __name__ == "__main__":
    main()