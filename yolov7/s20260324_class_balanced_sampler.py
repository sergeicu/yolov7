"""
Class-Balanced Samplers for YOLO Pediatric Wrist Fracture Detection
===================================================================

Custom PyTorch samplers for handling extreme class imbalance in the 4-class
bone-specific fracture detection problem:
    Class 0: distal_radius_fx
    Class 1: distal_ulna_fx
    Class 2: scaphoid_fx
    Class 3: ulna_styloid_fx

The samplers ensure roughly balanced representation across classes in each
training batch, addressing the severe imbalance where distal_radius dominates
(~73% of TPs) and scaphoid is extremely rare (~0.004% of TPs).

These samplers are designed to replace YOLOv7's default sampler during
training (Strategy 3: Class-Balanced Sampling from the training pipeline PRD).

Author: Claude Code
Date: 2026-03-26
"""

import os
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set


# =============================================================================
# Class Constants
# =============================================================================

CLASS_NAMES = {
    0: 'distal_radius_fx',
    1: 'distal_ulna_fx',
    2: 'scaphoid_fx',
    3: 'ulna_styloid_fx'
}

CLASS_IDS = set(range(4))


# =============================================================================
# Helper Functions
# =============================================================================

def parse_yolo_label(label_path: str) -> List[int]:
    """
    Parse a YOLO label file and return list of class IDs present.

    Args:
        label_path: Path to YOLO format label file (.txt)

    Returns:
        List of class IDs (integers 0-3) found in the label file.
        Returns empty list if file doesn't exist, is empty, or can't be parsed.
    """
    if not os.path.exists(label_path):
        return []

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        class_ids = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                class_id = int(parts[0])
                if 0 <= class_id <= 3:
                    class_ids.append(class_id)

        return class_ids
    except (ValueError, IOError, OSError):
        return []


def compute_class_frequencies_from_labels(
    label_dir: str,
    image_names: Optional[List[str]] = None
) -> Dict[int, int]:
    """
    Compute class frequency counts by scanning YOLO label files.

    Scans all label files in label_dir (or specified subset) and counts
    how many times each class (0-3) appears across all labels.

    Args:
        label_dir: Directory containing YOLO label .txt files
        image_names: Optional list of specific image names (without extension)
                     to process. If None, processes all .txt files in label_dir.

    Returns:
        Dictionary mapping class_id -> frequency count (number of boxes)
        Example: {0: 15000, 1: 3000, 2: 100, 3: 2500}

    Note:
        For multi-label images (multiple fractures in one image), each
        fracture box contributes to the count for its class.
    """
    class_counts = {i: 0 for i in range(4)}

    if image_names is not None:
        # Process only specified images
        for img_name in image_names:
            label_path = os.path.join(label_dir, f"{img_name}.txt")
            class_ids = parse_yolo_label(label_path)
            for cid in class_ids:
                class_counts[cid] += 1
    else:
        # Process all .txt files in directory
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(label_dir, filename)
                class_ids = parse_yolo_label(label_path)
                for cid in class_ids:
                    class_counts[cid] += 1

    return class_counts


def build_class_to_images_mapping(
    label_dir: str,
    image_names: List[str]
) -> Dict[int, List[int]]:
    """
    Build a mapping from class ID to list of image indices that contain
    at least one box of that class.

    This is the primary data structure used by ClassBalancedSampler to
    know which images to sample from for each class.

    Args:
        label_dir: Directory containing YOLO label .txt files
        image_names: List of image names (without extension) corresponding
                     to dataset indices

    Returns:
        Dictionary mapping class_id -> list of dataset indices (int)
        Example: {0: [0, 5, 10, ...], 1: [2, 7, ...], 2: [15, ...], 3: [3, 8, ...]}
    """
    class_to_indices = {i: [] for i in range(4)}

    for idx, img_name in enumerate(image_names):
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        class_ids = parse_yolo_label(label_path)

        # Track which classes this image has (set to avoid duplicates from same class)
        classes_in_image = set(class_ids)

        for cid in classes_in_image:
            if 0 <= cid <= 3:
                class_to_indices[cid].append(idx)

    return class_to_indices


def compute_inverse_frequency_weights(
    class_counts: Dict[int, int],
    epsilon: float = 1e-8
) -> Dict[int, float]:
    """
    Compute inverse frequency weights for each class.

    Weights are proportional to 1/frequency, normalized so that the
    minimum weight is 1.0. This ensures rare classes get higher weight.

    Args:
        class_counts: Dictionary mapping class_id -> count
        epsilon: Small value to avoid division by zero

    Returns:
        Dictionary mapping class_id -> weight
        Example: {0: 1.0, 1: 5.0, 2: 150.0, 3: 6.0} for severe imbalance
    """
    max_count = max(class_counts.values()) if class_counts else 1

    weights = {}
    for cid in range(4):
        count = class_counts.get(cid, 0)
        if count == 0:
            weights[cid] = float('inf')  # Infinite weight for unseen classes
        else:
            weights[cid] = max_count / (count + epsilon)

    # Normalize so minimum weight is 1.0
    min_weight = min(w for w in weights.values() if w != float('inf'))
    if min_weight > 0 and min_weight != 1.0:
        for cid in weights:
            if weights[cid] != float('inf'):
                weights[cid] /= min_weight

    return weights


# =============================================================================
# ClassBalancedSampler
# =============================================================================

class ClassBalancedSampler(Sampler):
    """
    A sampler that ensures roughly balanced class representation in each batch.

    This sampler is designed for the pediatric wrist fracture detection problem
    where:
    - Class 0 (distal_radius) dominates (~73% of positive boxes)
    - Class 2 (scaphoid) is extremely rare (~0.004% of positive boxes)

    The sampler works by:
    1. Building per-class lists of image indices
    2. In each epoch, samples images to fill batches with specified class ratios
    3. Ensuring all classes appear in every batch (or as close as possible)

    Args:
        class_to_indices: Dictionary mapping class_id -> list of dataset indices
                          that contain at least one box of that class.
                          Built using build_class_to_images_mapping().
        batch_size: Number of images per batch.
        target_ratios: Dictionary mapping class_id -> target fraction of batch
                      that should contain this class.
                      Default: equal distribution {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
                      For imbalanced data, might use: {0: 0.15, 1: 0.25, 2: 0.35, 3: 0.25}
        drop_last: If True, drop the last incomplete batch if it wouldn't
                   have enough samples to fill it.
        shuffle: If True, shuffle the sampling order each epoch.
        seed: Random seed for reproducibility.

    Example:
        >>> label_dir = "/path/to/train/labels"
        >>> image_names = [f.replace(".txt", "") for f in os.listdir(label_dir) if f.endswith(".txt")]
        >>> class_to_indices = build_class_to_images_mapping(label_dir, image_names)
        >>>
        >>> sampler = ClassBalancedSampler(
        ...     class_to_indices=class_to_indices,
        ...     batch_size=16,
        ...     target_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},  # Equal
        ...     drop_last=True,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>>
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
    """

    def __init__(
        self,
        class_to_indices: Dict[int, List[int]],
        batch_size: int,
        target_ratios: Optional[Dict[int, float]] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        if target_ratios is None:
            # Default: equal distribution across 4 classes
            target_ratios = {i: 0.25 for i in range(4)}

        # Validate target_ratios
        total_ratio = sum(target_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"target_ratios must sum to 1.0, got {total_ratio}")

        for cid in range(4):
            if cid not in target_ratios:
                raise ValueError(f"target_ratios must include class {cid}")

        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.target_ratios = target_ratios
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Compute how many samples per class per batch
        self.samples_per_class = {
            cid: max(1, int(batch_size * ratio))
            for cid, ratio in target_ratios.items()
        }

        # Total number of unique images in the dataset
        all_indices = set()
        for indices in class_to_indices.values():
            all_indices.update(indices)
        self.num_samples = len(all_indices)

        # Initialize random state
        self.rng = np.random.RandomState(seed)

        # Build epoch index list
        self._indices = None
        self.epoch = 0

    def __iter__(self):
        """Generate the sequence of sample indices for this epoch."""
        self.epoch = getattr(self, 'epoch', 0)
        seed = self.seed + self.epoch
        rng = np.random.RandomState(seed)

        # For tracking which images have been used this epoch
        available = {cid: list(indices.copy()) for cid, indices in self.class_to_indices.items()}
        all_used = set()

        # If shuffling, shuffle within each class's list
        if self.shuffle:
            for cid in available:
                rng.shuffle(available[cid])

        # Build batch index list
        batch_indices = []

        while True:
            # Check if we should stop
            remaining = self.num_samples - len(all_used)

            if self.drop_last:
                # Need at least batch_size samples remaining
                if remaining < self.batch_size:
                    break
            else:
                # Need at least 1 sample remaining
                if remaining == 0:
                    break

            # Build one batch
            batch = []
            classes_in_batch = set()

            for cid in range(4):
                # Target number of this class in batch
                target_n = self.samples_per_class[cid]

                # How many do we still need (accounting for duplicates)?
                # We want unique images in the batch
                needed = target_n

                # Pull from available pool for this class
                while needed > 0 and available[cid]:
                    idx = available[cid].pop()

                    # If we've already used this image in this epoch, skip
                    if idx in all_used:
                        continue

                    batch.append(idx)
                    all_used.add(idx)
                    classes_in_batch.add(cid)
                    needed -= 1

                # If we exhausted available images for this class, recycle
                if needed > 0 and not available[cid]:
                    # Put all images back (they're already in all_used, but we need more)
                    # This happens when a class has very few images
                    available[cid] = list(self.class_to_indices[cid].copy())
                    if self.shuffle:
                        rng.shuffle(available[cid])

            # Fill remaining slots in batch with random images if needed
            # (this can happen when some classes are exhausted)
            while len(batch) < self.batch_size:
                # Get an image from any class that still has available images
                found = False
                for cid in range(4):
                    if available[cid]:
                        idx = available[cid].pop()
                        if idx not in all_used:
                            batch.append(idx)
                            all_used.add(idx)
                            classes_in_batch.add(cid)
                            found = True
                            break

                if not found:
                    # All images have been used, exit loop
                    break

            if len(batch) > 0:
                if self.shuffle:
                    rng.shuffle(batch)
                batch_indices.extend(batch)
            else:
                break

        # Trim to exact multiple of batch_size if drop_last
        if self.drop_last and len(batch_indices) % self.batch_size != 0:
            batch_indices = batch_indices[:-(len(batch_indices) % self.batch_size)]

        return iter(batch_indices)

    def __len__(self):
        """Return the total number of samples that will be generated."""
        if self.drop_last:
            return (self.num_samples // self.batch_size) * self.batch_size
        else:
            return self.num_samples

    def set_epoch(self, epoch):
        """
        Set the epoch number for reproducible shuffling.

        Call this before each epoch starts (e.g., in DistributedDataParallel
        training where multiple processes need different shuffles).

        Args:
            epoch: The epoch number (int)
        """
        self.epoch = epoch


# =============================================================================
# WeightedMultiLabelSampler
# =============================================================================

class WeightedMultiLabelSampler(Sampler):
    """
    A weighted sampler for multi-label datasets that handles class imbalance.

    This sampler is an alternative to ClassBalancedSampler that assigns
    weights to each image based on the classes it contains. Images containing
    rare classes get higher weight, causing them to be sampled more frequently.

    This is useful when:
    - Images can contain multiple labels (multi-label problem)
    - You want smoother class balance than ClassBalancedSampler provides
    - You need more fine-grained control over sampling weights

    Args:
        class_to_indices: Dictionary mapping class_id -> list of dataset indices
                          that contain at least one box of that class.
        class_counts: Dictionary mapping class_id -> total count of boxes
                      for that class across the entire dataset.
                      Used to compute inverse frequency weights.
        num_samples: Total number of images in the dataset.
        batch_size: Number of images per batch.
        samples_per_batch: How many images to sample per batch. Default: batch_size.
                          Can be set lower than batch_size if using custom collate.
        drop_last: If True, drop the last incomplete batch.
        weight_exponent: Exponent to apply to inverse frequency weights.
                        Higher values = more aggressive balancing.
                        Default: 1.0 (standard inverse frequency).
        seed: Random seed for reproducibility.

    Example:
        >>> label_dir = "/path/to/train/labels"
        >>> image_names = [f.replace(".txt", "") for f in os.listdir(label_dir) if f.endswith(".txt")]
        >>> class_to_indices = build_class_to_images_mapping(label_dir, image_names)
        >>> class_counts = compute_class_frequencies_from_labels(label_dir, image_names)
        >>>
        >>> sampler = WeightedMultiLabelSampler(
        ...     class_to_indices=class_to_indices,
        ...     class_counts=class_counts,
        ...     num_samples=len(image_names),
        ...     batch_size=16,
        ...     weight_exponent=1.5,
        ...     seed=42
        ... )
    """

    def __init__(
        self,
        class_to_indices: Dict[int, List[int]],
        class_counts: Dict[int, int],
        num_samples: int,
        batch_size: int,
        samples_per_batch: Optional[int] = None,
        drop_last: bool = False,
        weight_exponent: float = 1.0,
        seed: int = 42
    ):
        self.class_to_indices = class_to_indices
        self.class_counts = class_counts
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.samples_per_batch = samples_per_batch if samples_per_batch else batch_size
        self.drop_last = drop_last
        self.weight_exponent = weight_exponent
        self.seed = seed

        # Compute inverse frequency weights
        self.class_weights = compute_inverse_frequency_weights(class_counts)

        # Build per-sample weight list
        # Each image's weight is the sum of its class weights
        self.sample_weights = self._compute_sample_weights()

        # Initialize random state
        self.rng = np.random.RandomState(seed)

        # Build indices list
        self._indices = None
        self.epoch = 0

    def _compute_sample_weights(self) -> List[float]:
        """Compute weight for each sample based on its class composition."""
        sample_weights = [0.0] * self.num_samples

        for cid, indices in self.class_to_indices.items():
            class_weight = self.class_weights.get(cid, 1.0) ** self.weight_exponent

            for idx in indices:
                # Add class weight (summing for multi-label images)
                sample_weights[idx] += class_weight

        # Normalize weights to sum to num_samples (so effective sample count is preserved)
        total_weight = sum(sample_weights)
        if total_weight > 0:
            normalization = self.num_samples / total_weight
            sample_weights = [w * normalization for w in sample_weights]

        return sample_weights

    def __iter__(self):
        """Generate weighted random sample indices for this epoch."""
        self.epoch = getattr(self, 'epoch', 0)
        seed = self.seed + self.epoch
        rng = np.random.RandomState(seed)

        # Generate weighted samples
        indices = list(range(self.num_samples))
        weights = self.sample_weights.copy()

        if self.shuffle:
            # For weighted sampling without replacement, we use rejection sampling
            # or simply shuffle and then rearrange based on weights
            rng.shuffle(indices)

        # Build batch list
        batch_indices = []
        sampled_in_epoch = set()

        while len(sampled_in_epoch) < self.num_samples:
            # Check if we should stop
            remaining = self.num_samples - len(sampled_in_epoch)

            if self.drop_last:
                if remaining < self.samples_per_batch:
                    break
            else:
                if remaining == 0:
                    break

            # Sample indices for this batch using weighted selection
            batch = []

            # Get unsampled indices and their weights
            unsampled_indices = [i for i in indices if i not in sampled_in_epoch]
            unsampled_weights = [weights[i] for i in unsampled_indices]

            if not unsampled_indices:
                break

            # Normalize weights for this selection round
            total_w = sum(unsampled_weights)
            if total_w > 0:
                probs = [w / total_w for w in unsampled_weights]
            else:
                probs = [1.0 / len(unsampled_indices)] * len(unsampled_indices)

            # Sample with replacement for weighted selection
            # (multiple passes through data with weighting)
            n_to_sample = min(self.samples_per_batch, len(unsampled_indices))

            try:
                # Use numpy choice for weighted sampling
                sampled = rng.choice(
                    unsampled_indices,
                    size=n_to_sample,
                    replace=False,
                    p=probs
                )
                batch = list(sampled)
            except ValueError:
                # Fallback to uniform sampling if weighted fails
                batch = unsampled_indices[:n_to_sample]

            batch_indices.extend(batch)
            sampled_in_epoch.update(batch)

        # Trim to exact multiple if drop_last
        if self.drop_last and len(batch_indices) % self.samples_per_batch != 0:
            batch_indices = batch_indices[:-(len(batch_indices) % self.samples_per_batch)]

        return iter(batch_indices)

    def __len__(self):
        """Return the total number of samples that will be generated."""
        if self.drop_last:
            return (self.num_samples // self.samples_per_batch) * self.samples_per_batch
        else:
            return self.num_samples

    @property
    def shuffle(self):
        """Whether to shuffle samples each epoch."""
        return True

    def set_epoch(self, epoch):
        """
        Set the epoch number for reproducible shuffling.

        Args:
            epoch: The epoch number (int)
        """
        self.epoch = epoch


# =============================================================================
# Utility: Create Sampler from Dataset Directory
# =============================================================================

def create_class_balanced_sampler_from_dirs(
    label_dir: str,
    batch_size: int,
    target_ratios: Optional[Dict[int, float]] = None,
    drop_last: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    return_stats: bool = False
) -> Tuple[ClassBalancedSampler, Dict]:
    """
    Convenience function to create a ClassBalancedSampler from a label directory.

    This function:
    1. Scans the label directory for all .txt files
    2. Computes class frequencies
    3. Builds the class-to-indices mapping
    4. Creates and returns the sampler

    Args:
        label_dir: Directory containing YOLO label .txt files
        batch_size: Number of images per batch
        target_ratios: Optional dict of class_id -> target ratio
        drop_last: Whether to drop incomplete final batch
        shuffle: Whether to shuffle each epoch
        seed: Random seed
        return_stats: If True, also return class statistics dict

    Returns:
        Tuple of (ClassBalancedSampler, stats_dict) if return_stats=True,
        else just ClassBalancedSampler

    Stats dict contains:
        - class_counts: frequency of each class
        - class_weights: inverse frequency weights
        - num_images: total number of images
        - class_distribution: per-class image counts

    Example:
        >>> sampler, stats = create_class_balanced_sampler_from_dirs(
        ...     label_dir="/path/to/train/labels",
        ...     batch_size=16,
        ...     target_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        ...     return_stats=True
        ... )
        >>> print(f"Class counts: {stats['class_counts']}")
    """
    # Get all image names from label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    image_names = [f.replace('.txt', '') for f in label_files]

    # Build mappings
    class_to_indices = build_class_to_images_mapping(label_dir, image_names)
    class_counts = compute_class_frequencies_from_labels(label_dir, image_names)

    # Create sampler
    sampler = ClassBalancedSampler(
        class_to_indices=class_to_indices,
        batch_size=batch_size,
        target_ratios=target_ratios,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed
    )

    if return_stats:
        class_weights = compute_inverse_frequency_weights(class_counts)
        class_distribution = {
            cid: len(indices) for cid, indices in class_to_indices.items()
        }

        stats = {
            'class_counts': class_counts,
            'class_weights': class_weights,
            'num_images': len(image_names),
            'class_distribution': class_distribution
        }

        return sampler, stats

    return sampler


# =============================================================================
# Example Usage with YOLOv7
# =============================================================================

"""
Example: Integrating ClassBalancedSampler with YOLOv7 Training
================================================================

Below is an example of how to integrate the ClassBalancedSampler into
a YOLOv7 training loop. This shows the pattern for modifying train_0118.py
or creating a wrapper script.

Prerequisites:
- Dataset has been prepared with s20260324_build_dataset.py
- Label files are in YOLO format with 4-class IDs (0-3)
- Training images are accessible (via symlinks or copies)

# -------------------------------------------------------------------------
# Option 1: Direct modification of train_0118.py
# -------------------------------------------------------------------------

# In train_0118.py, around line 300 where DataLoader is created,
# replace the default sampler with ClassBalancedSampler:

import sys
sys.path.insert(0, '/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolo_llm/experiments')
from s20260324_class_balanced_sampler import (
    ClassBalancedSampler,
    create_class_balanced_sampler_from_dirs,
    build_class_to_images_mapping,
    compute_class_frequencies_from_labels
)

# After dataset is created (around line 250 in train_0118.py):
# Build the sampler from the training labels directory
train_label_dir = os.path.join(data_path, 'train', 'labels')

# Create sampler with equal target ratios
sampler, stats = create_class_balanced_sampler_from_dirs(
    label_dir=train_label_dir,
    batch_size=batch_size,
    target_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},  # Equal balance
    drop_last=True,
    shuffle=True,
    seed=42,
    return_stats=True
)

print("Class-balanced sampler statistics:")
print(f"  Class counts: {stats['class_counts']}")
print(f"  Class weights: {stats['class_weights']}")
print(f"  Total training images: {stats['num_images']}")

# When creating the DataLoader, use our sampler as the batch_sampler
# Note: When using a batch_sampler, don't pass batch_size or sampler to DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,  # Use our custom batch sampler
    num_workers=workers,
    pin_memory=True,
    collate_fn=user_collate_fn  # YOLOv7's custom collate
)

# -------------------------------------------------------------------------
# Option 2: Custom training loop with sampler
# -------------------------------------------------------------------------

from torch.utils.data import DataLoader

# Define your dataset (adapt to YOLOv7's dataset class)
class YOLO4ClassDataset:
    def __init__(self, image_dir, label_dir, image_names):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        # Load image and label as YOLOv7 does...
        return image, label, _, _

# Setup
image_dir = "/path/to/dataset/train/images"
label_dir = "/path/to/dataset/train/labels"
image_names = [f.replace('.png', '') for f in os.listdir(image_dir) if f.endswith('.png')]

dataset = YOLO4ClassDataset(image_dir, label_dir, image_names)

# Create sampler
# For scaphoid-focused balancing (Strategy 3 experiments):
# Scaphoid at 25% of batch, others split remaining 75%
target_ratios = {
    0: 0.20,  # distal_radius: 20%
    1: 0.25,  # distal_ulna: 25%
    2: 0.30,  # scaphoid: 30% (oversample rare class)
    3: 0.25   # ulna_styloid: 25%
}

sampler, stats = create_class_balanced_sampler_from_dirs(
    label_dir=label_dir,
    batch_size=16,
    target_ratios=target_ratios,
    drop_last=True,
    shuffle=True,
    seed=42,
    return_stats=True
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=8,
    pin_memory=True
)

# Training loop
model = ...  # YOLOv7 model
optimizer = ...

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for distributed training reproducibility

    for batch_idx, (images, labels, paths, shapes) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = model(images, labels)
        loss.backward()
        optimizer.step()

# -------------------------------------------------------------------------
# Option 3: Using WeightedMultiLabelSampler (alternative approach)
# -------------------------------------------------------------------------

from s20260324_class_balanced_sampler import WeightedMultiLabelSampler

# Build mappings
class_to_indices = build_class_to_images_mapping(label_dir, image_names)
class_counts = compute_class_frequencies_from_labels(label_dir, image_names)

sampler = WeightedMultiLabelSampler(
    class_to_indices=class_to_indices,
    class_counts=class_counts,
    num_samples=len(image_names),
    batch_size=16,
    weight_exponent=1.5,  # More aggressive balancing
    drop_last=True,
    seed=42
)

dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=8)

# -------------------------------------------------------------------------
# YOLOv7 train_0118.py Integration Notes
# -------------------------------------------------------------------------
# 1. The sampler is set AFTER the dataset is created (line ~250)
# 2. The DataLoader is created at line ~300
# 3. Replace the default sampler with our ClassBalancedSampler
# 4. The collate_fn should remain YOLOv7's user_collate_fn
# 5. If using DistributedDataParallel, call sampler.set_epoch(epoch)
#    in the training loop before each epoch
#
# Key lines to modify in train_0118.py:
#   Before: dataset = LoadImagesAndLabels(...)  # line ~250
#   After:  sampler, stats = create_class_balanced_sampler_from_dirs(...)
#          print(f"Class distribution: {stats['class_distribution']}")
#
#   Before: dataloader = DataLoader(dataset, batch_size=batch_size, ...)
#   After:  dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
#
# -------------------------------------------------------------------------
# Testing the sampler standalone
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    # Test with a small synthetic dataset
    label_dir = "/path/to/your/label/directory"
    batch_size = 16

    sampler, stats = create_class_balanced_sampler_from_dirs(
        label_dir=label_dir,
        batch_size=batch_size,
        target_ratios={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        drop_last=True,
        shuffle=True,
        seed=42,
        return_stats=True
    )

    print(f"Number of samples per epoch: {len(sampler)}")
    print(f"Number of batches per epoch: {len(sampler) // batch_size}")
    print(f"Class counts: {stats['class_counts']}")
    print(f"Class weights: {stats['class_weights']}")
    print(f"Class distribution (images per class): {stats['class_distribution']}")

    # Iterate through one epoch
    print("\\nFirst 5 batches:")
    for i, batch_indices in enumerate(sampler):
        if i >= 5:
            break
        print(f"  Batch {i}: {batch_indices[:8]}... (length={len(batch_indices)})")
"""


if __name__ == "__main__":
    # Example usage when run as a script
    print("Class-Balanced Sampler for YOLO Pediatric Wrist Fracture Detection")
    print("=" * 70)
    print()
    print("This module provides three main classes:")
    print()
    print("1. ClassBalancedSampler:")
    print("   - Ensures each batch has roughly balanced class representation")
    print("   - Configurable target ratios per class")
    print("   - Best for extreme class imbalance (like scaphoid)")
    print()
    print("2. WeightedMultiLabelSampler:")
    print("   - Assigns weights based on inverse class frequency")
    print("   - Smoother balancing than ClassBalancedSampler")
    print("   - Good for multi-label images with varying class counts")
    print()
    print("3. Helper functions:")
    print("   - compute_class_frequencies_from_labels(): Count boxes per class")
    print("   - build_class_to_images_mapping(): Map classes to image indices")
    print("   - create_class_balanced_sampler_from_dirs(): One-step sampler creation")
    print()
    print("See the module docstring for integration examples with YOLOv7.")
