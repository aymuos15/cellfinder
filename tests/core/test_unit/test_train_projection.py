import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import pytest
import torch

from cellfinder.core.train.train_yaml import _reduce_z_collate


def _sample(planes):
    """(batch=1, y=1, x=1, z=N, c=1) cube from per-plane scalars, + label."""
    t = torch.tensor(planes, dtype=torch.float32).reshape(
        1, 1, 1, len(planes), 1
    )
    return t, torch.tensor([0, 1])


def test_center_keeps_middle_plane():
    data, _ = _reduce_z_collate(
        _sample([10.0, 20.0, 30.0]), z_axis=3, z_reduce="center"
    )
    assert data.shape == (1, 1, 1, 1)
    assert data.flatten().item() == 20.0


def test_max_projects_planes():
    data, _ = _reduce_z_collate(
        _sample([10.0, 30.0, 20.0]), z_axis=3, z_reduce="max"
    )
    assert data.shape == (1, 1, 1, 1)
    assert data.flatten().item() == 30.0


def test_mean_projects_planes():
    data, _ = _reduce_z_collate(
        _sample([10.0, 20.0, 30.0]), z_axis=3, z_reduce="mean"
    )
    assert data.flatten().item() == pytest.approx(20.0)


def test_bad_reduce_raises():
    with pytest.raises(ValueError, match="z_reduce must be"):
        _reduce_z_collate(_sample([1.0, 2.0]), z_axis=3, z_reduce="median")
