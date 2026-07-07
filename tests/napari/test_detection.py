from inspect import signature
from unittest.mock import patch

import napari
import pytest

from cellfinder.core.main import main as cellfinder_run
from cellfinder.napari.detect import detect_widget
from cellfinder.napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder.napari.detect.thread_worker import Worker
from cellfinder.napari.sample_data import load_sample


def test_data_inputs_dimensions_passed():
    """The 2D/3D selector defaults to 3D and reaches the core entry point."""
    args = DataInputs(
        signal_array=None, background_array=None
    ).as_core_arguments()
    assert args["dimensions"] == 3
    assert set(args) <= set(signature(cellfinder_run).parameters)


@pytest.fixture
def get_detect_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = detect_widget()
    for layer in load_sample():
        viewer.add_layer(napari.layers.Image(layer[0], **layer[1]))
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder", widget_name="Cell detection"
    )
    return widget


def test_detect_worker():
    """
    Smoke test to check that the detection worker runs
    """
    data = load_sample()
    signal = data[0][0]
    background = data[1][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(),
        ClassificationInputs(trained_model=None),
        MiscInputs(start_plane=0, end_plane=1),
    )
    worker.work()


@pytest.mark.parametrize(
    argnames="analyse_local",
    argvalues=[True, False],  # increase test coverage by covering both cases
)
def test_run_detect(get_detect_widget, analyse_local):
    """
    Test backend is called
    """
    with patch("cellfinder.napari.detect.detect.Worker") as worker:
        get_detect_widget.analyse_local.value = analyse_local
        get_detect_widget.call_button.clicked()
        assert worker.called


def test_run_detect_without_inputs():
    """ """
    with patch("cellfinder.napari.detect.detect.show_info") as show_info:
        widget = (
            detect_widget()
        )  # won't have image layers, so should notice and show info
        widget.call_button.clicked()
        assert show_info.called


def test_reset_defaults(get_detect_widget):
    """Smoke test that restore defaults doesn't error."""
    get_detect_widget.reset_button.clicked()
