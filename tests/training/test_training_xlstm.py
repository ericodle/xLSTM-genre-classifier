import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
class TestTrainingxLSTM:
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent.parent.absolute()

    @pytest.fixture
    def test_data_dir(self):
        return "outputs/test-mfcc-extraction/mfccs_splits"

    @pytest.fixture
    def output_base(self):
        return tempfile.mkdtemp(prefix="test_training_xlstm_")

    def test_train_xlstm(self, test_data_dir, output_base, project_root):
        output_dir = os.path.join(output_base, "xlstm_test")

        cmd = [
            "python",
            "src/training/train_model.py",
            "--data",
            test_data_dir,
            "--model",
            "xLSTM",
            "--output",
            output_dir,
            "--epochs",
            "3",
            "--batch-size",
            "8",
            "--lr",
            "0.001",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 0, f"xLSTM training failed:\n{result.stderr}"
        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
