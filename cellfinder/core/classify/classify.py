import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from torch.utils.data import DataLoader

from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import (
    CuboidArrayDataset,
    CuboidBatchSampler,
)
from cellfinder.core.classify.tools import get_model, model_input_channels
from cellfinder.core.tools.image_processing import dataset_mean_std
from cellfinder.core.tools.tools import (
    deprecate_positional_args,
    ensure_3d,
    validate_central_planes,
    validate_dimensions,
)
from cellfinder.core.train.train_yaml import depth_type, models


def _classify_2d_batch(model, data, z_axis, infer_z_planes, infer_pool):
    """Run a 2D model on a cube batch and return per-candidate predictions.

    For single-plane inference the depth-1 z axis is squeezed. For 2.5D
    inference (``infer_z_planes`` > 1) the central z-planes are folded into the
    batch, scored independently by the same 2D model, then pooled per
    candidate: ``mean`` averages the softmax, ``vote`` averages the per-plane
    one-hot decisions (a majority).
    """
    n_planes = data.shape[z_axis]
    if infer_z_planes == 1:
        if n_planes != 1:
            raise ValueError(
                "2D classification expects a depth-1 cube, but got depth "
                f"{n_planes}"
            )
        return model(data.squeeze(z_axis))

    batch = data.shape[0]
    folded = data.movedim(z_axis, 1)
    planes = folded.reshape(batch * n_planes, *folded.shape[2:])
    per_plane = model(planes).reshape(batch, n_planes, -1)
    if infer_pool == "vote":
        votes = torch.nn.functional.one_hot(
            per_plane.argmax(dim=-1), per_plane.shape[-1]
        ).to(per_plane.dtype)
        return votes.mean(dim=1)
    return per_plane.mean(dim=1)


@deprecate_positional_args
def main(
    *,
    points: List[Cell],
    signal_array: types.array,
    background_array: Optional[types.array],
    n_free_cpus: int,
    voxel_sizes: Tuple[float, float, float],
    network_voxel_sizes: Tuple[float, float, float],
    batch_size: int,
    cube_height: int,
    cube_width: int,
    cube_depth: int,
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    network_depth: depth_type,
    max_workers: int = 3,
    pin_memory: bool = False,
    dimensions: int = 3,
    infer_z_planes: int = 1,
    infer_pool: str = "mean",
    callback: Optional[Callable[[int], None]] = None,
    normalize_channels: bool = False,
    normalization_n_sampling_planes: int = 50,
) -> List[Cell]:
    """
    Parameters
    ----------

    points: List of Cell objects
        The potential cells to classify.
    signal_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    background_array : numpy.ndarray or dask array, optional
        3D array representing the background data in z, y, x order. If
        ``None``, a single-channel (signal-only) cube is built and a
        single-channel model must be used.
    n_free_cpus : int
        How many CPU cores to leave free.
    voxel_sizes : 3-tuple of floats
        Size of your voxels in the z, y, and x dimensions.
    network_voxel_sizes : 3-tuple of floats
        Size of the pre-trained network's voxels in the z, y, and x dimensions.
    batch_size : int
        How many potential cells to classify at one time. The GPU/CPU
        memory must be able to contain at once this many data cubes for
        the models. For performance-critical applications, tune to maximize
        memory usage without running out. Check your GPU/CPU memory to verify
        it's not full.
    cube_height: int
        The height of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_width: int
        The width of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_depth: int
        The depth of the data cube centered on the cell used for
        classification. Defaults to `20`.
    trained_model : Optional[Path]
        Trained model file path (home directory (default) -> pretrained
        weights).
    model_weights : Optional[Path]
        Model weights path (home directory (default) -> pretrained
        weights).
    network_depth: str
        The network depth to use during classification. Defaults to `"50"`.
    max_workers: int
        The max number of sub-processes to use for data loading / processing.
        Defaults to 8.
    pin_memory: bool
        Pins data to be sent to the GPU to the CPU memory. This allows faster
        GPU data speeds, but can only be used if the data used by the GPU can
        stay in the CPU RAM while the GPU uses it. I.e. there's enough RAM.
        Otherwise, if there's a risk of the RAM being paged, it shouldn't be
        used. Defaults to False.
    dimensions: int
        Whether to classify using a 3D network (a z-stack cube, the default)
        or a 2D network (a single plane). When 2, 2D signal/background arrays
        are accepted and a depth-1 cube is squeezed to a 2D (y, x, c) image.
    infer_z_planes: int
        2.5D inference. When greater than 1 (2D networks only), the 2D model
        is run on this many central z-planes per candidate and the per-plane
        predictions are pooled. Defaults to 1 (single-plane 2D inference). No
        retraining is needed; any single-plane 2D model can be used.
    infer_pool: str
        How per-plane predictions are pooled when ``infer_z_planes`` > 1.
        ``"mean"`` averages the softmax probabilities (the default and
        recommended); ``"vote"`` takes a majority of per-plane decisions.
    callback : Callable[int], optional
        A callback function that is called during classification. Called with
        the batch number once that batch has been classified.
    normalize_channels : bool
        If True, the signal and background data will be each normalized
        to a mean of zero and standard deviation of 1. Defaults to False.
    normalization_n_sampling_planes : int
        If `normalize_channels` is True, the data arrays will be down-sampled
        in the first axis to use approximately this many planes -- equally
        spaced, before calculating their mean/std. E.g. a value of 50 for a
        dataset of 200 planes means every fourth plane will be used. Defaults
        to 50.
    """
    validate_dimensions(dimensions)
    validate_central_planes(infer_z_planes, dimensions, "2.5D inference")
    if infer_z_planes > 1 and infer_pool not in ("mean", "vote"):
        raise ValueError(
            f"infer_pool must be 'mean' or 'vote', got {infer_pool!r}"
        )

    signal_array = ensure_3d(
        signal_array, dimensions, name="Signal data", error=IOError
    )
    if background_array is not None:
        background_array = ensure_3d(
            background_array, dimensions, name="Background data", error=IOError
        )
    if dimensions == 2:
        if len(voxel_sizes) == 2:
            voxel_sizes = (1.0, *voxel_sizes)
        if len(network_voxel_sizes) == 2:
            network_voxel_sizes = (1.0, *network_voxel_sizes)
        # the depth-1 cube must not be rescaled in z, so match the z voxel
        # size to the data and pull a single plane
        network_voxel_sizes = (voxel_sizes[0], *network_voxel_sizes[1:])
        cube_depth = infer_z_planes

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(min_free_cpu_cores=n_free_cpus)
    workers = min(workers, max_workers)

    start_time = datetime.now()

    voxel_sizes = list(map(float, voxel_sizes))

    signal_normalization = background_normalization = None
    if normalize_channels:
        logger.debug("Calculating channels norms")
        signal_normalization = dataset_mean_std(
            signal_array, normalization_n_sampling_planes
        )
        background_normalization = dataset_mean_std(
            background_array, normalization_n_sampling_planes
        )
        logger.debug(
            f"Signal channel norm is: {signal_normalization}. "
            f"Background channel norm is: {background_normalization}"
        )

    logger.debug("Initialising cube generator")
    dataset = CuboidArrayDataset(
        signal_array=signal_array,
        background_array=background_array,
        signal_normalization=signal_normalization,
        background_normalization=background_normalization,
        points=points,
        data_voxel_sizes=voxel_sizes,
        network_voxel_sizes=network_voxel_sizes,
        network_cuboid_voxels=(cube_depth, cube_height, cube_width),
        axis_order=("z", "y", "x"),
        max_axis_0_cuboids_buffered=1,
    )
    # we use our own sampler so we can control the ordering
    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sort_by_axis="z",
        auto_shuffle=False,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    if trained_model and Path(trained_model).suffix == ".h5":
        logger.warning(
            "Weights provided in place of the model, "
            "loading weights into default model."
        )
        model_weights = trained_model
        trained_model = None

    model_shape = None
    if dimensions == 2:
        model_shape = (cube_height, cube_width, dataset.num_channels)
    model = get_model(
        existing_model=trained_model,
        model_weights=model_weights,
        network_depth=models[network_depth],
        inference=True,
        num_channels=dataset.num_channels,
        dimensions=dimensions,
        shape=model_shape,
    )

    model_channels = model_input_channels(model)
    if model_channels != dataset.num_channels:
        raise ValueError(
            f"The classification model expects {model_channels}-channel "
            f"input but {dataset.num_channels} channel(s) were provided. "
            f"Use a `trained_model` whose channel count matches the data, "
            f"or provide data matching the model (signal only for 1, "
            f"signal + background for 2)."
        )

    logger.info("Running inference")
    z_axis = dataset.output_axis_order.index("z") + 1
    if workers:
        dataset.start_dataset_thread(workers)
    try:
        outputs = []
        for step, data in tqdm.tqdm(
            enumerate(data_loader),
            total=len(sampler),
            desc="Classifying",
            unit="batches",
            smoothing=1 / (25 * max(1, workers)),
            mininterval=0.5,
        ):
            if dimensions == 2:
                output = _classify_2d_batch(
                    model, data, z_axis, infer_z_planes, infer_pool
                )
            else:
                output = model(data)
            # in original keras, it seemed to held on to the output until the
            # end (possibly on GPU). This causes resources issues for very
            # heavy loads. So instead immediately move it to cpu/numpy
            outputs.append(output.cpu().numpy())
            if callback is not None:
                callback(step)
    finally:
        dataset.stop_dataset_thread()

    predictions = np.argmax(np.concatenate(outputs, axis=0), axis=1)
    points_list = []

    # only go through the "extractable" points
    k = 0
    # the sampler doesn't auto shuffle, so the classification order (i.e. order
    # in `predictions`) is the same order as the sampler returns the batches.
    # Use that to get the corresponding row in points_arr, which gives us the
    # `index` of the row in the original point in the input points list
    for arr in sampler:
        for i in arr:
            p_idx = int(dataset.points_arr[i, 4].item())
            # don't use the original cell, use a copy
            cell = deepcopy(points[p_idx])
            cell.type = int((predictions[k] + 1).item())
            points_list.append(cell)
            k += 1

    time_elapsed = datetime.now() - start_time
    logger.info(
        f"Classification complete - {len(points_list)} points "
        f"done in : {time_elapsed}"
    )

    return points_list
