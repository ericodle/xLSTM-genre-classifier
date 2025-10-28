import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
class TestTrainingConventionalML:
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent.parent.absolute()

    @pytest.fixture
    def test_data_dir(self):
        return "outputs/test-mfcc-extraction/mfccs_splits"

    @pytest.fixture
    def output_base(self):
        return tempfile.mkdtemp(prefix="test_training_conventional_")

    @pytest.mark.parametrize("model_name", ["svm", "rf", "nb", "knn"])
    def test_train_conventional_model(self, model_name, test_data_dir, output_base, project_root):
        output_dir = os.path.join(output_base, f"{model_name}_test")

        cmd = [
            "python",
            "src/training/train_conventional_ml.py",
            "--data",
            test_data_dir,
            "--model",
            model_name,
            "--output",
            output_dir,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 0, f"Conventional ML training failed:\n{result.stderr}"
        assert os.path.exists(os.path.join(output_dir, f"{model_name}.joblib"))
        assert os.path.exists(os.path.join(output_dir, "confusion_matrix.png"))
