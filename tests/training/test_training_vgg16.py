import os
import subprocess
import tempfile

import pytest


@pytest.mark.slow
class TestTrainingVGG16:
    @pytest.fixture
    def test_data_dir(self):
        return "outputs/test-mfcc-extraction/mfccs_splits"

    @pytest.fixture
    def output_base(self):
        return tempfile.mkdtemp(prefix="test_training_vgg16_")

    def test_train_vgg16(self, test_data_dir, output_base):
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

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )

        assert result.returncode == 0, f"VGG16 training failed:\n{result.stderr}"
        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
