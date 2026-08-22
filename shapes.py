"""Synthetic geometric shapes generator for 2D image and 3D volume autoencoder demos."""

import torch

VALID_SIZES = (64, 128, 256)
SHAPE_CLASSES_2D = ("circle", "square", "triangle", "cross")
SHAPE_CLASSES_3D = ("sphere", "cube", "cylinder", "cone")


def _grid_2d(size):
    coords = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _grid_3d(size):
    coords = torch.linspace(-1, 1, size)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return xx, yy, zz


def _draw_2d_shape(shape, size, scale, center):
    xx, yy = _grid_2d(size)
    xx = xx - center[0]
    yy = yy - center[1]
    half = scale
    if shape == "circle":
        mask = xx ** 2 + yy ** 2 <= half ** 2
    elif shape == "square":
        mask = (xx.abs() <= half) & (yy.abs() <= half)
    elif shape == "triangle":
        mask = (yy >= -half) & (yy <= half) & (xx.abs() <= half * (1 - (yy + half) / (2 * half)))
    elif shape == "cross":
        thickness = half * 0.35
        mask = ((xx.abs() <= thickness) | (yy.abs() <= thickness)) & (xx.abs() <= half) & (yy.abs() <= half)
    else:
        raise ValueError(f"Unknown 2D shape: {shape}")
    return mask.float()


def _draw_3d_shape(shape, size, scale, center):
    xx, yy, zz = _grid_3d(size)
    xx = xx - center[0]
    yy = yy - center[1]
    zz = zz - center[2]
    half = scale
    if shape == "sphere":
        mask = xx ** 2 + yy ** 2 + zz ** 2 <= half ** 2
    elif shape == "cube":
        mask = (xx.abs() <= half) & (yy.abs() <= half) & (zz.abs() <= half)
    elif shape == "cylinder":
        mask = (xx ** 2 + yy ** 2 <= half ** 2) & (zz.abs() <= half)
    elif shape == "cone":
        mask = (zz >= -half) & (zz <= half) & (xx ** 2 + yy ** 2 <= (half * (1 - (zz + half) / (2 * half))) ** 2)
    else:
        raise ValueError(f"Unknown 3D shape: {shape}")
    return mask.float()


def generate_2d_shapes(n_samples, size=64, n_classes=4, seed=None):
    """Generate a synthetic 2D shapes dataset: returns (images, labels).

    `size` only controls the image resolution (height=width=size); it is a separate
    knob from the autoencoder's `depth` and must stay independent of it.
    """
    # TODO (student): restrict `size` to one of VALID_SIZES (64, 128 or 256) and raise
    # a clear error otherwise. Do not derive `size` from `depth` - keep them decoupled.
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    shapes = SHAPE_CLASSES_2D[:n_classes]

    images = []
    labels = []
    for i in range(n_samples):
        label = i % len(shapes)
        scale = 0.4 + 0.3 * torch.rand(1, generator=generator).item()
        center = 0.3 * (torch.rand(2, generator=generator) - 0.5) * 2
        mask = _draw_2d_shape(shapes[label], size, scale, (center[0].item(), center[1].item()))
        images.append(mask.unsqueeze(0))
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def generate_3d_shapes(n_samples, size=64, n_classes=4, seed=None):
    """Generate a synthetic 3D shapes dataset: returns (volumes, labels).

    `size` only controls the volume resolution (depth=height=width=size); it is a
    separate knob from the autoencoder's `depth` and must stay independent of it.
    """
    # TODO (student): restrict `size` to one of VALID_SIZES (64, 128 or 256) and raise
    # a clear error otherwise. Do not derive `size` from `depth` - keep them decoupled.
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    shapes = SHAPE_CLASSES_3D[:n_classes]

    volumes = []
    labels = []
    for i in range(n_samples):
        label = i % len(shapes)
        scale = 0.4 + 0.3 * torch.rand(1, generator=generator).item()
        center = 0.2 * (torch.rand(3, generator=generator) - 0.5) * 2
        mask = _draw_3d_shape(shapes[label], size, scale, tuple(center.tolist()))
        volumes.append(mask.unsqueeze(0))
        labels.append(label)

    volumes = torch.stack(volumes)
    labels = torch.tensor(labels, dtype=torch.long)
    return volumes, labels
