import os
import subprocess
import tempfile

import pytest


@pytest.mark.slow
class TestTrainingTransformers:
    @pytest.fixture
    def test_data_dir(self):
        return "outputs/test-mfcc-extraction/mfccs_splits"

    @pytest.fixture
    def output_base(self):
        return tempfile.mkdtemp(prefix="test_training_transformers_")

    @pytest.mark.parametrize(
        "model_name,extra_args",
        [
            ("Transformer", ["--batch-size", "4", "--lr", "0.0001"]),
            ("ViT", ["--batch-size", "4", "--lr", "0.0001"]),
        ],
    )
    def test_train_transformer_like(self, model_name, extra_args, test_data_dir, output_base):
        output_dir = os.path.join(output_base, f"{model_name.lower()}_test")

        cmd = [
            "python",
            "src/training/train_model.py",
            "--data",
            test_data_dir,
            "--model",
            model_name,
            "--output",
            output_dir,
            "--epochs",
            "3",
        ] + extra_args

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )

        assert result.returncode == 0, f"{model_name} training failed:\n{result.stderr}"
        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
