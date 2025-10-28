#!/usr/bin/env python3
"""
Test all model architectures in the project.

This test suite ensures that:
1. All models can be instantiated correctly
2. Forward pass works with appropriate input shapes
3. Output shapes are correct
4. Models don't crash during inference
"""

import os
import sys

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    CNN_model,
    FC_model,
    GRU_model,
    LSTM_model,
    Transformer,
    VGG16Classifier,
    ViTClassifier,
    xLSTM,
)


class TestFC:
    """Test Fully Connected model."""

    def test_fc_model_instantiation(self):
        """Test FC model can be created."""
        model = FC_model(input_dim=100, output_dim=10, dropout=0.1)
        assert model is not None
        assert model.model_name == "FC_model"

    def test_fc_forward_pass_2d(self):
        """Test FC forward pass with 2D input."""
        model = FC_model(input_dim=100, output_dim=10)
        batch_size = 32

        # Input: (batch, features)
        x = torch.randn(batch_size, 100)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_fc_forward_pass_flattened(self):
        """Test FC forward pass with 3D input (auto-flattened)."""
        model = FC_model(input_dim=100, output_dim=10)
        batch_size = 32

        # Input: (batch, time, features) - should be flattened
        x = torch.randn(batch_size, 5, 20)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestCNN:
    """Test Convolutional Neural Network model."""

    def test_cnn_model_instantiation(self):
        """Test CNN model can be created."""
        model = CNN_model(num_classes=10, dropout=0.1)
        assert model is not None
        assert model.model_name == "CNN_model"

    def test_cnn_forward_pass_2d(self):
        """Test CNN forward pass with 2D input (MFCC flattened)."""
        # Use fewer layers and smaller features to avoid pooling collapse
        model = CNN_model(
            num_classes=10,
            conv_layers=2,  # Only 2 layers to avoid pooling issues
            base_filters=16,  # Smaller filters
        )
        batch_size = 16

        # Input: (batch, features) - flattened MFCC features
        # CNN reshapes this to (batch, 1, 1, features)
        # Use a size that won't collapse during pooling
        # With 2 conv layers and pooling every 2nd block, need at least 4 spatial dims
        x = torch.randn(batch_size, 200)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_cnn_forward_pass_3d(self):
        """Test CNN forward pass with 3D MFCC-like input."""
        model = CNN_model(num_classes=10)
        batch_size = 16

        # Input: (batch, time, features) - typical MFCC shape
        x = torch.randn(batch_size, 50, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_cnn_regression_mode(self):
        """Test CNN in regression mode outputs membership scores."""
        model = CNN_model(num_classes=10, regression_mode=True)
        batch_size = 16
        x = torch.randn(batch_size, 50, 13)

        output = model(x)

        # In regression mode, output should be in [0, 1] range
        assert output.shape == (batch_size, 10)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
        assert not torch.isnan(output).any()


class TestLSTM:
    """Test Long Short-Term Memory model."""

    def test_lstm_model_instantiation(self):
        """Test LSTM model can be created."""
        model = LSTM_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        assert model is not None
        assert model.model_name == "LSTM_model"

    def test_lstm_forward_pass_3d(self):
        """Test LSTM forward pass with 3D input."""
        model = LSTM_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        batch_size = 16

        # Input: (batch, time, features) - typical for RNNs
        x = torch.randn(batch_size, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_lstm_forward_pass_2d(self):
        """Test LSTM forward pass with 2D input (will add sequence dim)."""
        model = LSTM_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        batch_size = 16

        # Input: (batch, features) - will add sequence dimension
        x = torch.randn(batch_size, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_lstm_multiple_layers(self):
        """Test LSTM with multiple layers."""
        model = LSTM_model(
            input_dim=13, hidden_dim=32, layer_dim=2, output_dim=10, dropout_prob=0.1  # 2 layers
        )
        batch_size = 16
        x = torch.randn(batch_size, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestGRU:
    """Test Gated Recurrent Unit model."""

    def test_gru_model_instantiation(self):
        """Test GRU model can be created."""
        model = GRU_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        assert model is not None
        assert model.model_name == "GRU_model"

    def test_gru_forward_pass_3d(self):
        """Test GRU forward pass with 3D input."""
        model = GRU_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        batch_size = 16
        x = torch.randn(batch_size, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_gru_forward_pass_2d(self):
        """Test GRU forward pass with 2D input."""
        model = GRU_model(
            input_dim=13,
            hidden_dim=32,
            layer_dim=2,  # Use 2 layers to allow dropout
            output_dim=10,
            dropout_prob=0.1,
        )
        batch_size = 16
        x = torch.randn(batch_size, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestTransformer:
    """Test Transformer model."""

    def test_transformer_model_instantiation(self):
        """Test Transformer model can be created."""
        model = Transformer(
            input_dim=13,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            output_dim=10,
            dropout=0.1,
        )
        assert model is not None
        assert model.model_name == "Transformer"

    def test_transformer_forward_pass_3d(self):
        """Test Transformer forward pass with 3D input."""
        model = Transformer(
            input_dim=13,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            output_dim=10,
            dropout=0.1,
        )
        batch_size = 16
        x = torch.randn(batch_size, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestXLSTM:
    """Test xLSTM model."""

    def test_xlstm_model_instantiation(self):
        """Test xLSTM model can be created."""
        model = xLSTM(input_dim=13, hidden_dim=32, num_layers=1, output_dim=10, dropout=0.1)
        assert model is not None
        assert model.model_name == "xLSTM"

    def test_xlstm_forward_pass_3d(self):
        """Test xLSTM forward pass with 3D input."""
        model = xLSTM(input_dim=13, hidden_dim=32, num_layers=1, output_dim=10, dropout=0.1)
        batch_size = 16
        x = torch.randn(batch_size, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestVGG16:
    """Test VGG16 model."""

    def test_vgg16_model_instantiation_pretrained(self):
        """Test VGG16 with pretrained weights."""
        model = VGG16Classifier(num_classes=10, pretrained=True, dropout=0.1, num_mfcc_features=13)
        assert model is not None
        assert "VGG16" in model.model_name

    def test_vgg16_model_instantiation_scratch(self):
        """Test VGG16 from scratch."""
        model = VGG16Classifier(num_classes=10, pretrained=False, dropout=0.1, num_mfcc_features=13)
        assert model is not None

    def test_vgg16_forward_pass(self):
        """Test VGG16 forward pass with MFCC input."""
        model = VGG16Classifier(num_classes=10, pretrained=False, dropout=0.1, num_mfcc_features=13)
        batch_size = 8
        # VGG16 expects 4D input: (batch, channels, height, width)
        # VGG16 uses asymmetric pooling (2,1) to preserve feature width
        # With 5 pooling layers, we need: input_height >= 2^5 = 32
        # Use a realistic MFCC size: ~1300 time steps (30s @ 22050Hz)
        x = torch.randn(batch_size, 1, 1300, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestViT:
    """Test Vision Transformer model."""

    def test_vit_model_instantiation_pretrained(self):
        """Test ViT with pretrained weights."""
        model = ViTClassifier(num_classes=10, pretrained=True, dropout=0.1, num_mfcc_features=13)
        assert model is not None
        assert "ViT" in model.model_name

    def test_vit_model_instantiation_scratch(self):
        """Test ViT from scratch."""
        model = ViTClassifier(num_classes=10, pretrained=False, dropout=0.1, num_mfcc_features=13)
        assert model is not None

    def test_vit_forward_pass(self):
        """Test ViT forward pass with MFCC input."""
        model = ViTClassifier(num_classes=10, pretrained=False, dropout=0.1, num_mfcc_features=13)
        batch_size = 8
        # ViT expects 4D input: (batch, channels, height, width)
        # It will internally resize to 224x224
        x = torch.randn(batch_size, 1, 100, 13)
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestModelConsistency:
    """Test consistency across models."""

    def test_all_models_have_base_functionality(self):
        """Test that all models inherit from BaseModel and have common methods."""
        from src.models import BaseModel

        models = [
            FC_model(input_dim=100, output_dim=10),
            CNN_model(num_classes=10),
            LSTM_model(input_dim=13, hidden_dim=32, layer_dim=2, output_dim=10, dropout_prob=0.1),
            GRU_model(input_dim=13, hidden_dim=32, layer_dim=2, output_dim=10, dropout_prob=0.1),
            Transformer(
                input_dim=13,
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                ff_dim=128,
                output_dim=10,
                dropout=0.1,
            ),
            xLSTM(input_dim=13, hidden_dim=32, num_layers=2, output_dim=10, dropout=0.1),
        ]

        for model in models:
            assert isinstance(model, BaseModel)
            assert hasattr(model, "forward")
            assert hasattr(model, "model_config")
            assert hasattr(model, "model_name")
            assert hasattr(model, "is_trained")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
