"""
GAN-based data augmentation module for GenreDiscern.
"""

from .models import GanGenerator, GanDiscriminator
from .train_gan import GanTrainer
from .augment import GanAugmenter

__all__ = ["GanGenerator", "GanDiscriminator", "GanTrainer", "GanAugmenter"]

