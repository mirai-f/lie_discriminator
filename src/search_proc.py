import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from scipy.sparse import issparse
from src.preprocessor import get_features_names_out


class SearchProc:
    """
    Класс SearchProc выполняет:

    1. Поиск гиперпараметров с RandomizedSearchCV (self.fit_search_model)
    2. Обучение модели с лучшими параметрами (self.fit_best_model)
    3. Оценку качества (self.evaluate_model)
    4. Вывод лучших параметров (self.print_search_results)
    5. Вывод важнейших признаков (self.print_top_features)

    при запуске метода self.run_proc()
    """

    def __init__(self, model_class, param_grid, preprocessor, n_iter=100, scoring="f1", verbose=0):
        self.model_class = model_class
        self.param_grid = param_grid

        self.model = None
        self.search_model = None

        self.n_params = len(list(ParameterGrid(self.param_grid)))
        self.n_iter = n_iter if n_iter < self.n_params else self.n_params
        self.scoring = scoring
        self.verbose = verbose

        self.preprocessor = preprocessor

    def _create_pipeline(self):
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model_class())
        ])
        
    def _get_features_names_out(self, preprocessor):
        """Использована внешняя функция"""
        return np.array(get_features_names_out(preprocessor))

    def fit_search_model(self, X_train, y_train):
        print(f"Starting hyperparameter search ({self.n_iter} params)...")
        start_time = time.time()

        estimator = self._create_pipeline()
        self.search_model = RandomizedSearchCV(
            estimator,
            self.param_grid,
            n_iter=self.n_iter,
            cv=5,
            n_jobs=-1,
            verbose=self.verbose,
            scoring=self.scoring
        )

        self.search_model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"Hyperparameter search completed in {elapsed:.2f} seconds\n")

    def fit_best_model(self, X_train, y_train):
        print("Training best model...")
        start_time = time.time()

        best_params = self.search_model.best_params_
        self.model = self._create_pipeline()
        self.model.set_params(**best_params)
        self.model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        print(f"Best model trained in {elapsed:.2f} seconds\n")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred), "\n")

    def print_search_results(self, top_n=3):
        cv_results = (
            pd.DataFrame(self.search_model.cv_results_)
            .sort_values("mean_test_score", ascending=False)
            [["mean_fit_time", "params", "mean_test_score"]]
            .round({
                "mean_fit_time": 2,
                "mean_test_score": 4,
                "std_test_score": 4
            })
        )[:top_n]
        cv_results["params"] = cv_results["params"].apply(
            lambda x: "\n".join(f"{k}: {v}" for k, v in x.items())
        )

        print(f"Top {top_n} combinations:")
        print(tabulate(
            cv_results,
            headers=cv_results.columns,
            tablefmt="grid",
            stralign="left",
            numalign="center",
            showindex=False
        ))
        print()

    def print_top_features(self, top_n=5):
        model = self.model["model"]

        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            print("This model does not support feature importance extraction.")
            return

        features_names = self._get_features_names_out(self.preprocessor)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        
        if issparse(importances):
            importances = importances.toarray().flatten()

        print(f"Top {top_n} most important features:")
        indices = np.argsort(np.abs(importances))[::-1][:top_n]
        max_word_len = max(len(features_names[i]) for i in indices)
        for i in indices:
            print(f"{features_names[i]:<{max_word_len}} : {importances[i]:+.4f}")

    def run_proc(self, X_train, X_test, y_train, y_test):
        self.fit_search_model(X_train, y_train)
        self.fit_best_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.print_search_results()
        self.print_top_features()