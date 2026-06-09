import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest
import torch
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.classify import classify
from cellfinder.core.classify.classify import _classify_2d_batch
from cellfinder.core.classify.resnet import build_model


def _cube(planes):
    """Build a (batch=1, y=1, x=1, z=N, c=1) cube from per-plane scalars."""
    t = torch.tensor(planes, dtype=torch.float32)
    return t.reshape(1, 1, 1, len(planes), 1)


def _model_returns(per_plane_probs):
    """A fake 2D model returning preset (rows, 2) probabilities in order."""
    table = torch.tensor(per_plane_probs, dtype=torch.float32)

    def model(planes):
        assert planes.shape[0] == table.shape[0]
        return table

    return model


def test_single_plane_squeezes_and_calls_model():
    data = _cube([7.0])  # depth 1
    captured = {}

    def model(x):
        captured["shape"] = tuple(x.shape)
        return torch.tensor([[0.2, 0.8]])

    out = _classify_2d_batch(
        model, data, z_axis=3, infer_z_planes=1, infer_pool="mean"
    )
    # z axis dropped -> (batch, y, x, c)
    assert captured["shape"] == (1, 1, 1, 1)
    assert out.shape == (1, 2)


def test_single_plane_rejects_multi_depth():
    data = _cube([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="depth-1 cube"):
        _classify_2d_batch(
            lambda x: x, data, z_axis=3, infer_z_planes=1, infer_pool="mean"
        )


def test_mean_pool_averages_softmax():
    data = _cube([1.0, 2.0, 3.0])  # 3 planes
    model = _model_returns([[0.0, 1.0], [1.0, 0.0], [0.4, 0.6]])
    out = _classify_2d_batch(
        model, data, z_axis=3, infer_z_planes=3, infer_pool="mean"
    )
    expected = torch.tensor([[(0.0 + 1.0 + 0.4) / 3, (1.0 + 0.0 + 0.6) / 3]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_vote_pool_takes_majority():
    data = _cube([1.0, 2.0, 3.0])
    # plane argmaxes: class1, class0, class1 -> majority class1
    model = _model_returns([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    out = _classify_2d_batch(
        model, data, z_axis=3, infer_z_planes=3, infer_pool="vote"
    )
    assert out.argmax(dim=-1).item() == 1
    # two of three planes voted class1
    assert torch.allclose(out, torch.tensor([[1 / 3, 2 / 3]]), atol=1e-6)


@pytest.mark.parametrize(
    "infer_z_planes,infer_pool",
    [(1, "mean"), (3, "mean"), (3, "vote")],
    ids=["single-plane", "mp3-mean", "mp3-vote"],
)
def test_classify_2d_and_2p5d_end_to_end(
    synthetic_single_spot, tmp_path, infer_z_planes, infer_pool
):
    """2D single-plane and 2.5D multi-plane classification run end to end
    with a real 2D model and cube generator."""
    signal_array, _background, c_xyz = synthetic_single_spot
    signal_array = signal_array.astype(np.uint16)
    points = [Cell(tuple(int(c) for c in c_xyz), Cell.UNKNOWN)]

    model = build_model(
        shape=(50, 50, 1), network_depth="18-layer", dimensions=2
    )
    model_path = tmp_path / "model_2d_1ch.keras"
    model.save(model_path)

    result = classify.main(
        points=points,
        signal_array=signal_array,
        background_array=None,
        n_free_cpus=0,
        voxel_sizes=(5, 1, 1),
        network_voxel_sizes=(5, 1, 1),
        batch_size=1,
        cube_height=50,
        cube_width=50,
        cube_depth=20,
        trained_model=model_path,
        model_weights=None,
        network_depth="18",
        dimensions=2,
        infer_z_planes=infer_z_planes,
        infer_pool=infer_pool,
    )

    assert len(result) == len(points)
    assert all(cell.type in (Cell.CELL, Cell.NO_CELL) for cell in result)
