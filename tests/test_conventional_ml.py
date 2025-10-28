"""
Tests for conventional machine learning models.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.conventional_ml import (
    GaussianNBModel,
    KNNModel,
    RandomForestModel,
    SVMModel,
    get_conventional_model,
)


# Generate synthetic test data
@pytest.fixture
def sample_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class TestSVM:
    """Test Support Vector Machine model."""

    def test_svm_instantiation(self):
        """Test SVM model can be created."""
        model = SVMModel(kernel="rbf", C=1.0)
        assert model is not None
        assert model.model_name == "SVM-rbf"
        assert not model.is_trained

    def test_svm_fit_and_predict(self, sample_data):
        """Test SVM training and prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = SVMModel(kernel="linear", C=1.0)

        model.fit(X_train, y_train)
        assert model.is_trained

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert len(np.unique(predictions)) <= 5  # Should be class labels

    def test_svm_with_different_kernels(self, sample_data):
        """Test SVM with different kernels."""
        X_train, X_test, y_train, y_test = sample_data

        for kernel in ["linear", "rbf", "poly"]:
            model = SVMModel(kernel=kernel)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert predictions.shape == y_test.shape


class TestRandomForest:
    """Test Random Forest model."""

    def test_random_forest_instantiation(self):
        """Test Random Forest model can be created."""
        model = RandomForestModel(n_estimators=50)
        assert model is not None
        assert model.model_name == "RandomForest"
        assert not model.is_trained

    def test_random_forest_fit_and_predict(self, sample_data):
        """Test Random Forest training and prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = RandomForestModel(n_estimators=50, random_state=42)

        model.fit(X_train, y_train)
        assert model.is_trained

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        # Random Forest supports predict_proba
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 5)  # 5 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_random_forest_with_different_params(self, sample_data):
        """Test Random Forest with different parameters."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestModel(
            n_estimators=30, max_depth=5, min_samples_split=5, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape


class TestNaiveBayes:
    """Test Gaussian Naive Bayes model."""

    def test_naive_bayes_instantiation(self):
        """Test Naive Bayes model can be created."""
        model = GaussianNBModel()
        assert model is not None
        assert model.model_name == "GaussianNB"
        assert not model.is_trained

    def test_naive_bayes_fit_and_predict(self, sample_data):
        """Test Naive Bayes training and prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = GaussianNBModel()

        model.fit(X_train, y_train)
        assert model.is_trained

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        # Naive Bayes supports predict_proba
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 5)  # 5 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestKNN:
    """Test K-Nearest Neighbors model."""

    def test_knn_instantiation(self):
        """Test KNN model can be created."""
        model = KNNModel(n_neighbors=5)
        assert model is not None
        assert model.model_name == "KNN"
        assert not model.is_trained

    def test_knn_fit_and_predict(self, sample_data):
        """Test KNN training and prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = KNNModel(n_neighbors=3)

        model.fit(X_train, y_train)
        assert model.is_trained

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        # KNN supports predict_proba
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 5)  # 5 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_knn_different_neighbors(self, sample_data):
        """Test KNN with different k values."""
        X_train, X_test, y_train, y_test = sample_data

        for k in [1, 3, 5, 7]:
            model = KNNModel(n_neighbors=k)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert predictions.shape == y_test.shape


class TestModelFactory:
    """Test model factory function."""

    def test_get_svm_model(self):
        """Test factory creates SVM."""
        model = get_conventional_model("svm", kernel="rbf", C=1.0)
        assert isinstance(model, SVMModel)
        assert model.model_name == "SVM-rbf"

    def test_get_random_forest_model(self):
        """Test factory creates Random Forest."""
        model = get_conventional_model("rf", n_estimators=50)
        assert isinstance(model, RandomForestModel)
        assert model.model_name == "RandomForest"

    def test_get_naive_bayes_model(self):
        """Test factory creates Naive Bayes."""
        model = get_conventional_model("nb")
        assert isinstance(model, GaussianNBModel)
        assert model.model_name == "GaussianNB"

    def test_get_knn_model(self):
        """Test factory creates KNN."""
        model = get_conventional_model("knn", n_neighbors=5)
        assert isinstance(model, KNNModel)
        assert model.model_name == "KNN"

    def test_invalid_model_type(self):
        """Test factory raises error for invalid type."""
        with pytest.raises(ValueError):
            get_conventional_model("invalid_type")


class TestModelInfo:
    """Test model information methods."""

    def test_get_model_info(self, sample_data):
        """Test model info retrieval."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestModel(n_estimators=50)
        info = model.get_model_info()

        assert info["model_name"] == "RandomForest"
        assert info["model_type"] == "RandomForestModel"
        assert info["is_trained"] == False
        assert "model_config" in info

        model.fit(X_train, y_train)
        info = model.get_model_info()
        assert info["is_trained"] == True


class TestErrorHandling:
    """Test error handling."""

    def test_predict_before_training(self, sample_data):
        """Test prediction before training raises error."""
        X_train, X_test, y_train, y_test = sample_data

        model = SVMModel()
        with pytest.raises(RuntimeError, match="must be trained before prediction"):
            model.predict(X_test)

    def test_proba_for_svm(self, sample_data):
        """Test that linear SVM doesn't support predict_proba by default."""
        X_train, X_test, y_train, y_test = sample_data

        model = SVMModel(kernel="linear")
        model.fit(X_train, y_train)

        # LinearSVC doesn't have predict_proba by default
        # This should raise an error or return decision function
        with pytest.raises(AttributeError):
            model.predict_proba(X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
