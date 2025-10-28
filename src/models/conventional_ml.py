"""
Conventional machine learning models for GenreDiscern.
Includes SVM, Random Forest, Naive Bayes, and KNN classifiers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


class TraditionalMLBase(ABC):
    """Abstract base class for traditional ML models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.model_config: Dict[str, Any] = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying sklearn model."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        if self.model is None:
            self.model = self._create_model()

        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before prediction")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_name} does not support predict_proba")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "model_config": self.model_config,
        }


class SVMModel(TraditionalMLBase):
    """Support Vector Machine classifier."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        degree: int = 3,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        super().__init__(f"SVM-{kernel}")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.scale_features = scale_features

        self.model_config = {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "degree": degree,
            "random_state": random_state,
            "scale_features": scale_features,
        }

    def _create_model(self):
        """Create SVM model with optional scaling."""
        if self.kernel == "linear":
            svm = LinearSVC(C=self.C, random_state=self.random_state)
        else:
            svm = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                random_state=self.random_state,
            )

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("svm", svm)])
        return svm


class RandomForestModel(TraditionalMLBase):
    """Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        scale_features: bool = False,
    ):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.scale_features = scale_features

        self.model_config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "scale_features": scale_features,
        }

    def _create_model(self):
        """Create Random Forest model."""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("rf", rf)])
        return rf


class GaussianNBModel(TraditionalMLBase):
    """Gaussian Naive Bayes classifier."""

    def __init__(self, var_smoothing: float = 1e-9, scale_features: bool = False):
        super().__init__("GaussianNB")
        self.var_smoothing = var_smoothing
        self.scale_features = scale_features

        self.model_config = {
            "var_smoothing": var_smoothing,
            "scale_features": scale_features,
        }

    def _create_model(self):
        """Create Gaussian Naive Bayes model."""
        nb = GaussianNB(var_smoothing=self.var_smoothing)

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("nb", nb)])
        return nb


class KNNModel(TraditionalMLBase):
    """K-Nearest Neighbors classifier."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        scale_features: bool = True,
    ):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.scale_features = scale_features

        self.model_config = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "scale_features": scale_features,
        }

    def _create_model(self):
        """Create KNN model."""
        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
        )

        if self.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("knn", knn)])
        return knn


def get_conventional_model(model_type: str, **kwargs):
    """
    Factory function to create conventional ML models.

    Args:
        model_type: Type of model ('svm', 'rf', 'nb', 'knn')
        **kwargs: Model-specific parameters

    Returns:
        TraditionalMLBase instance
    """
    model_type = model_type.lower()

    if model_type in ["svm", "support_vector_machine"]:
        return SVMModel(**kwargs)
    elif model_type in ["rf", "random_forest"]:
        return RandomForestModel(**kwargs)
    elif model_type in ["nb", "naive_bayes", "gaussian_nb"]:
        return GaussianNBModel(**kwargs)
    elif model_type in ["knn", "k_nearest_neighbors"]:
        return KNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
