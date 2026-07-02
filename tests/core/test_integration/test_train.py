import os
import sys

import keras
import pytest
from pytest_mock.plugin import MockerFixture

from cellfinder.core.classify.tools import model_input_channels
from cellfinder.core.train.train_yaml import cli as train_run
from cellfinder.core.train.train_yaml import run as train_run_fn

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "training"
)
cell_cubes = os.path.join(data_dir, "cells")
non_cell_cubes = os.path.join(data_dir, "non_cells")
training_yaml_file = os.path.join(data_dir, "training.yaml")
training_yaml_single_channel = os.path.join(
    data_dir, "training_single_channel.yaml"
)
training_yaml_file_stats = os.path.join(data_dir, "training_with_stats.yaml")


EPOCHS = "2"

# only checks that the model is trained, and then saved.
# doesn't check that it works etc


@pytest.mark.slow
def test_train(mocker, tmpdir):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    mocker.patch("sys.argv", train_args)

    train_run()

    model_file = os.path.join(tmpdir, "model.keras")
    assert os.path.exists(model_file)


def test_projection_rejected_in_3d(tmp_path):
    with pytest.raises(ValueError, match="only valid for 2D"):
        train_run_fn(
            str(tmp_path),
            training_yaml_file,
            dimensions=3,
            z_planes=3,
        )


@pytest.mark.slow
def test_train_2d_projection(tmp_path):
    import keras
    import tifffile
    from brainglobe_utils.IO.yaml import save_yaml

    # build depth-3 cubes from the central planes of the 3D test cubes, so the
    # max-projection over --z-planes 3 has real planes to reduce
    cubes = tmp_path / "cells"
    cubes.mkdir()
    for fname in os.listdir(cell_cubes):
        if fname.endswith(".tif"):
            cube = tifffile.imread(os.path.join(cell_cubes, fname))
            tifffile.imwrite(cubes / fname, cube[9:12])

    yaml_file = tmp_path / "training_2d.yaml"
    save_yaml(
        {
            "data": [
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes),
                    "signal_channel": 0,
                    "type": "cell",
                },
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes),
                    "signal_channel": 0,
                    "type": "no_cell",
                },
            ]
        },
        yaml_file,
    )

    out_dir = str(tmp_path / "out")
    sys.argv = [
        "cellfinder_train",
        "-y",
        str(yaml_file),
        "-o",
        out_dir,
        "--epochs",
        "1",
        "--dimensions",
        "2",
        "--z-planes",
        "3",
        "--z-reduce",
        "max",
        "--no-augment",
    ]
    train_run()

    model_file = os.path.join(out_dir, "model.keras")
    assert os.path.exists(model_file)
    model = keras.models.load_model(model_file)
    # projection collapses z, so the model is still a 2D (batch, y, x, c) net
    assert len(model.input_shape) == 4


def test_train_2d_projection_normalized(mocker: MockerFixture, tmp_path):
    import tifffile
    from brainglobe_utils.IO.yaml import save_yaml

    # depth-3 cubes so max-projection over --z-planes 3 has planes to reduce,
    # plus per-channel stats so --normalize-channels has data to apply
    cubes = tmp_path / "cells"
    cubes.mkdir()
    for fname in os.listdir(cell_cubes):
        if fname.endswith(".tif"):
            cube = tifffile.imread(os.path.join(cell_cubes, fname))
            tifffile.imwrite(cubes / fname, cube[9:12])

    stats = {
        "signal_mean": 241.31,
        "signal_std": 154.92,
        "bg_mean": 650.94,
        "bg_std": 217.90,
    }
    yaml_file = tmp_path / "training_2d_norm.yaml"
    save_yaml(
        {
            "data": [
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes),
                    "signal_channel": 0,
                    "type": "cell",
                    **stats,
                },
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes),
                    "signal_channel": 0,
                    "type": "no_cell",
                    **stats,
                },
            ]
        },
        yaml_file,
    )

    out_dir = str(tmp_path / "out")
    mocker.patch(
        "sys.argv",
        [
            "cellfinder_train",
            "-y",
            str(yaml_file),
            "-o",
            out_dir,
            "--epochs",
            "1",
            "--dimensions",
            "2",
            "--z-planes",
            "3",
            "--z-reduce",
            "max",
            "--normalize-channels",
            "--no-augment",
        ],
    )
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )
    train_run()

    # normalization stats reached the 2D, z-reduced training and val datasets
    (fit_mock,) = [
        m for m in get_model.mock_calls if repr(m).startswith("call().fit(")
    ]
    train_ds = fit_mock.kwargs["x"].dataset
    val_ds = fit_mock.kwargs["validation_data"].dataset
    assert train_ds.points_norm_arr is not None
    assert val_ds.points_norm_arr is not None


@pytest.mark.slow
def test_train_single_channel(mocker, tmpdir):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_single_channel,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    mocker.patch("sys.argv", train_args)
    train_run()

    model_file = os.path.join(tmpdir, "model.keras")
    assert os.path.exists(model_file)

    model = keras.models.load_model(model_file)
    assert model_input_channels(model) == 1


@pytest.mark.slow
def test_train_2d(tmp_path):
    import tifffile
    from brainglobe_utils.IO.yaml import save_yaml

    # 2D training consumes depth-1 cubes (as produced by 2D curation), so
    # build them from the central plane of the existing 3D test cubes.
    cubes_2d = tmp_path / "cells"
    cubes_2d.mkdir()
    for fname in os.listdir(cell_cubes):
        if fname.endswith(".tif"):
            cube = tifffile.imread(os.path.join(cell_cubes, fname))
            tifffile.imwrite(cubes_2d / fname, cube[10:11])

    yaml_file = tmp_path / "training_2d.yaml"
    save_yaml(
        {
            "data": [
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes_2d),
                    "signal_channel": 0,
                    "type": "cell",
                },
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes_2d),
                    "signal_channel": 0,
                    "type": "no_cell",
                },
            ]
        },
        yaml_file,
    )

    out_dir = str(tmp_path / "out")
    train_args = [
        "cellfinder_train",
        "-y",
        str(yaml_file),
        "-o",
        out_dir,
        "--epochs",
        "1",
        "--dimensions",
        "2",
        "--no-augment",
    ]
    sys.argv = train_args
    train_run()

    model_file = os.path.join(out_dir, "model.keras")
    assert os.path.exists(model_file)

    import keras

    model = keras.models.load_model(model_file)
    # a 2D model takes batch + y + x + channels
    assert len(model.input_shape) == 4


@pytest.mark.parametrize("lr_schedule", [True, False])
def test_train_lr_schedule(mocker: MockerFixture, tmpdir, lr_schedule):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
        "--lr-multiplier",
        "0.3",
    ]
    if lr_schedule:
        train_args.extend(["--lr-schedule", "10", "20"])

    mocker.patch("sys.argv", train_args)
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )

    train_run()
    # get the data sets passed to fit(). There's no clear name property of
    # the mock fit call, so use its repr
    (fit_mock,) = [
        m for m in get_model.mock_calls if repr(m).startswith("call().fit(")
    ]
    callbacks = fit_mock.kwargs["callbacks"]

    # locate the scheduler callback, if any
    from keras.callbacks import LearningRateScheduler

    callbacks = [c for c in callbacks if isinstance(c, LearningRateScheduler)]
    if lr_schedule:
        assert len(callbacks) == 1
        # the callback is a partial function with these args
        partial_callback = callbacks[0].schedule
        assert partial_callback.keywords["multiplier"] == 0.3
        assert partial_callback.keywords["epoch_list"] == [10, 20]
    else:
        assert not callbacks


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("has_norms", [True, False])
def test_train_normalization_missing_stats(
    mocker: MockerFixture, tmpdir, has_norms, normalize
):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file_stats if has_norms else training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    if normalize:
        train_args.append("--normalize-channels")

    mocker.patch("sys.argv", train_args)
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )

    if normalize and not has_norms:
        # if the yaml doesn't have normalization info an error will be raised
        with pytest.raises(ValueError):
            train_run()
    else:
        train_run()
        # get the data sets passed to fit() to verify if it has norm data
        # there's no clear name property of the mock fit call, so use its repr
        (fit_mock,) = [
            m
            for m in get_model.mock_calls
            if repr(m).startswith("call().fit(")
        ]
        train_dataset = fit_mock.kwargs["x"].dataset
        val_dataset = fit_mock.kwargs["validation_data"].dataset

        if normalize:
            # if we normalize, the normalization data should be in dataset
            assert train_dataset.points_norm_arr is not None
            assert val_dataset.points_norm_arr is not None
        else:
            # otherwise, no normalization data should have been passed, even if
            # the yaml has it
            assert train_dataset.points_norm_arr is None
            assert val_dataset.points_norm_arr is None
