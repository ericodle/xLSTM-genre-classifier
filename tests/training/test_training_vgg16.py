import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
class TestTrainingVGG16:
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent.parent.absolute()

    @pytest.fixture
    def test_data_dir(self):
        return "outputs/test-mfcc-extraction/mfccs_splits"

    @pytest.fixture
    def output_base(self):
        return tempfile.mkdtemp(prefix="test_training_vgg16_")

    @pytest.mark.skip(reason="VGG16 is too slow even without pretrained weights")
    def test_train_vgg16(self, test_data_dir, output_base, project_root):
        output_dir = os.path.join(output_base, "vgg16_test")

        cmd = [
            "python",
            "src/training/train_model.py",
            "--data",
            test_data_dir,
            "--model",
            "VGG16",
            "--output",
            output_dir,
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--lr",
            "0.0005",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 0, f"VGG16 training failed:\n{result.stderr}"
        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
