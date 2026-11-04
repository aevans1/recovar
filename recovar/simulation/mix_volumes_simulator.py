"""Simulator for images from a mixture of principal volumes

Image i is simulated from mean_vol + sum_j z_ij pc_vols_j

Mean vol and pcs are imported from existing recovar pipeline

z coords are custom loaded (see make_test_dataset_from_PCs) 

Same imaging forward model and output writing as `simulator'
"""

import logging

import jax
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils
from recovar import core
from recovar.data_io import cryoem_dataset
from recovar.simulation import simulator

logger = logging.getLogger(__name__)


def load_volumes_from_pipeline(pipeline_dir, grid_size, n_pcs=2):
    """ Load mean and top PCs from a pipeline as centered fourier volumes. """
    from recovar.output import output

    po = output.PipelineOutput(pipeline_dir)
    volume_shape = tuple(int(s) for s in po.get("volume_shape"))
    if volume_shape != (grid_size,) * 3:
        raise ValueError(
            f"pipeline volume_shape {volume_shape} does not match grid_size={grid_size}. "
            "Resizing is not supported; use a pipeline run at the simulation grid size."
        )
    volumes = np.array(po.get_u(n_pcs=n_pcs), dtype=np.complex64)  # (n_pcs, N), Fourier
    if volumes.shape[0] != n_pcs:
        raise ValueError(
            f"Requested n_pcs={n_pcs} but source pipeline '{pipeline_dir}' provided "
            f"only {volumes.shape[0]} saved eigenvectors."
        )
    mean_volume = np.array(po.get("mean"), dtype=np.complex64).reshape(1, -1)  # (1, N), Fourier
    return mean_volume, volumes, float(po.get("voxel_size"))



def generate_synthetic_dataset_mix_volumes(
    output_folder,
    latent_space_bounds,
    voxel_size,
    source_pipeline_dir,
    n_images,
    outlier_file_input=None,
    grid_size=128,
    n_pcs=2,
    latent_distribution=None,
    image_assignments=None,
    dataset_params_option="dataset1",
    noise_level=1.0,
    noise_model="radial1",
    put_extra_particles=False,
    percent_outliers=0.0,
    volume_radius=0.9,
    trailing_zero_format_in_vol_name=True,
    noise_scale_std=0.3,
    contrast_std=0.3,
    disc_type="linear_interp",
    n_tilts=-1,
    dose_per_tilt=3,
    angle_per_tilt=3,
    image_dtype=np.float16,
    image_offset_n_std=0.0,
    per_particle_contrast=True,
    premultiplied_ctf=False,
    noise_increase_per_tilt=None,
    create_nested_structure=False,
    nested_prefix="Extract/job193",
    percent_tilt_series_outliers=0.0,
    noise_rng_batch_size=None,
):
    """
    noise_rng_batch_size : int, optional
        Batch size used only to advance the random-noise stream. When omitted,
        it matches the image processing batch size. Supplying a fixed value
        keeps generated noise independent of GPU-memory-driven processing
        batch changes (issue #148 fix).
    """
    from recovar.output import output

    output.mkdir_safe(output_folder)

    logger.info("Loading up mean and u's from some pipline")
    logger.info(
        "Note: for now, gridsize of simulation has to match that of the pipeline load stuff. in future these will be downsampled if needed"
    )
    mean_volume, pc_volumes, voxel_size = load_volumes_from_pipeline(source_pipeline_dir, grid_size, n_pcs=n_pcs)

    logger.info("getting mean of volumes and mean_volume to divide by their norm, hacky for now.")
    all_volumes = np.vstack([pc_volumes, mean_volume])

    scale_vol = 1 / np.mean(np.linalg.norm(all_volumes, axis=-1))
    pc_volumes = pc_volumes * scale_vol
    mean_volume = mean_volume * scale_vol
    outlier_volume = (
        fourier_transform_utils.get_dft3(utils.load_mrc(outlier_file_input)).reshape(-1)
        if outlier_file_input is not None
        else None
    )

    dataset_param_generator = simulator.get_pose_ctf_generator(dataset_params_option)
    noise_variance = simulator.get_noise_model(noise_model, grid_size) / 50000 * noise_level

    mrc_file = None

    rescale_noise = True
    if rescale_noise:
        # Calibrate on at most 10k images, using their own latents (avoids out-of-range indices for small n_images)
        n_cal = min(n_images, 10000)
        # Dont use premultiplied_ctf for
        main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, _ = (
            generate_simulated_dataset_mix_volumes(
                mean_volume,
                pc_volumes,
                latent_space_bounds,
                voxel_size,
                latent_distribution,
                image_assignments[:n_cal],
                n_cal,
                noise_variance,
                noise_scale_std,
                contrast_std,
                put_extra_particles,
                percent_outliers=percent_outliers,
                dataset_param_generator=dataset_param_generator,
                volume_radius=volume_radius,
                outlier_volume=outlier_volume,
                disc_type=disc_type,
                mrc_file=mrc_file,
                image_offset_n_std=image_offset_n_std,
                per_particle_contrast=per_particle_contrast,
                premultiplied_ctf=False,
            )
        )
        norm_image_square = np.mean(main_image_stack**2)
        norm_image = norm_image_square
        noise_variance = noise_variance / (norm_image)
        pc_volumes = pc_volumes / np.sqrt(norm_image)
        mean_volume = mean_volume / np.sqrt(norm_image)
        scale_vol = scale_vol / np.sqrt(norm_image)

    main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, tilt_groups = (
        generate_simulated_dataset_mix_volumes(
            mean_volume,
            pc_volumes,
            latent_space_bounds,
            voxel_size,
            latent_distribution,
            image_assignments,
            n_images,
            noise_variance,
            noise_scale_std,
            contrast_std,
            put_extra_particles,
            percent_outliers=percent_outliers,
            dataset_param_generator=dataset_param_generator,
            volume_radius=volume_radius,
            outlier_volume=outlier_volume,
            disc_type=disc_type,
            mrc_file=mrc_file,
            n_tilts=n_tilts,
            dose_per_tilt=dose_per_tilt,
            angle_per_tilt=angle_per_tilt,
            image_offset_n_std=image_offset_n_std,
            per_particle_contrast=per_particle_contrast,
            premultiplied_ctf=premultiplied_ctf,
            noise_increase_per_tilt=noise_increase_per_tilt,
            percent_tilt_series_outliers=percent_tilt_series_outliers,
            noise_rng_batch_size=noise_rng_batch_size,
        )
    )

    # Add additional simulation parameters that weren't set yet
    additional_params = {
        # Volume parameters
        "source_pipeline_dir": source_pipeline_dir,
        "scale_vol": scale_vol,
        # Dataset parameters
        "grid_size": grid_size,
        "dataset_params_option": dataset_params_option,
        "outlier_file_input": outlier_file_input,
        # Noise parameters
        "noise_model": noise_model,
        "noise_level": noise_level,
    }
    simulation_info.update(additional_params)

    simulator.save_simulated_dataset(
        output_folder,
        main_image_stack,
        ctf_params,
        rots,
        trans,
        simulation_info,
        voxel_size,
        grid_size,
        tilt_groups,
        image_dtype=image_dtype,
        create_nested_structure=create_nested_structure,
        nested_prefix=nested_prefix,
    )

    return main_image_stack, simulation_info


def generate_simulated_dataset_mix_volumes(
    mean_volume,
    pc_volumes,
    latent_space_bounds,
    voxel_size,
    latent_distribution,
    image_assignments,
    n_images,
    noise_variance,
    noise_scale_std,
    contrast_std,
    put_extra_particles,
    percent_outliers=0.0,
    dataset_param_generator=None,
    volume_radius=0.95,
    outlier_volume=None,
    disc_type="linear_interp",
    mrc_file=None,
    n_tilts=-1,
    dose_per_tilt=None,
    angle_per_tilt=None,
    voltage=100,
    image_offset_n_std=0.0,
    per_particle_contrast=True,
    premultiplied_ctf=False,
    noise_increase_per_tilt=None,
    percent_tilt_series_outliers=0.0,
    noise_rng_batch_size=None,
):

    volume_shape = utils.guess_vol_shape_from_vol_size(pc_volumes[0].size)
    grid_size = volume_shape[0]

    ctf_params, rots, trans = dataset_param_generator(n_images, grid_size)

    if "ewald" in disc_type:
        phase_shift = np.arcsin(ctf_params[:, core.CTFParamIndex.W]) / np.pi * 180
        ctf_params[:, core.CTFParamIndex.W] = 0
        ctf_params[:, core.CTFParamIndex.PHASE_SHIFT] = phase_shift
        ctf_params[:, core.volt_ind] = 100

    trans *= 0
    per_image_contrast, per_image_noise_scale = simulator.generate_contrast_params(n_images, noise_scale_std, contrast_std)

    # image_assignments are the preloaded (n_images, n_pcs) latent coordinates
    if n_tilts > 0:
        raise NotImplementedError("tilt series not supported yet for mix volume simulator")
    else:
        per_tilt_contrast = None
        tilt_groups = None
        tilt_series_assignment = None
        ctf_evaluator = core.CTFEvaluator()

    main_dataset = cryoem_dataset.CryoEMDataset(
        None,
        voxel_size,
        cryoem_dataset.ImageMetadata(rots, trans, ctf_params),
        ctf_evaluator=ctf_evaluator,
        grid_size=grid_size,
    )

    batch_size, noise_rng_batch_size = simulator.simulation_batch_sizes(grid_size, disc_type, noise_rng_batch_size)

    main_image_stack = simulate_data_mix_volumes(
        main_dataset,
        mean_volume,
        pc_volumes,
        noise_variance,
        batch_size,
        image_assignments,
        per_image_contrast,
        per_image_noise_scale,
        seed=0,
        disc_type=disc_type,
        mrc_file=mrc_file,
        premultiplied_ctf=premultiplied_ctf,
        noise_rng_batch_size=noise_rng_batch_size,
    )

    per_image_offset = simulator.add_image_offsets(main_image_stack, image_offset_n_std)

    ######## TODO: implement these if this simulator file ends up getting used going forward, for not necessary
    if put_extra_particles:
        raise NotImplementedError("extra particles not supported yet in mix volumes simulator")
    if percent_outliers > 0:
        raise NotImplementedError("percent outliers not supported yet in mix volumes simulator")
    #######

    simulation_info = {
        "ctf_params": ctf_params,
        "rots": rots,
        "trans": trans,
        "per_image_contrast": per_image_contrast,
        "per_image_noise_scale": per_image_noise_scale,
        "per_image_offset": per_image_offset,
        "image_assignment": image_assignments,
        "noise_variance": noise_variance.astype(np.float32),
        "voxel_size": voxel_size,
        "tilt_series_assignment": tilt_series_assignment,
        "tilt_groups": tilt_groups,
        "per_tilt_contrast": per_tilt_contrast,
        # Add noise-related parameters
        "noise_increase_per_tilt": noise_increase_per_tilt,
        "dose_indices": None,
        "dose_per_tilt": None,
        "angle_per_tilt": None,
        "n_tilts": None,
        "simulation_batch_size": batch_size,
        "simulation_noise_rng_batch_size": noise_rng_batch_size,
    }

    return main_image_stack, ctf_params, rots, trans, simulation_info, voxel_size, tilt_groups


def simulate_data_mix_volumes(
    experiment_dataset,
    mean_volume,
    pc_volumes,
    noise_variance,
    batch_size,
    image_assignments,
    per_image_contrast,
    per_image_noise_scale,
    seed=0,
    disc_type="linear_interp",
    mrc_file=None,
    pad_before_translate=False,
    Bfactor=100,
    premultiplied_ctf=False,
    noise_rng_batch_size=None,
):
    """Simulate images from mean_vol + sum_j z_:j pc_vols_j

    ``image_assignments`` is (n_images, n_pcs) array of latent coords z.
    At each batch, CTF-free projections of all vols are weighted by z and summed and
     then passed through the same CTF + noise step as `simulate_data` from original simulator.
    """
    key = jax.random.PRNGKey(seed)

    if noise_rng_batch_size is None:
        noise_rng_batch_size = batch_size
    noise_rng_batch_size = utils.safe_batch_size(noise_rng_batch_size)

    if mrc_file is None:
        output_array = np.zeros(
            [experiment_dataset.n_images, *experiment_dataset.image_shape], dtype=experiment_dataset.dtype_real
        )
    else:
        output_array = mrc_file.data

    n_images = experiment_dataset.n_images
    noise_subkeys = []
    for _ in range(0, int(np.ceil(n_images / noise_rng_batch_size))):
        key, subkey = jax.random.split(key)
        noise_subkeys.append(subkey)

    # The mean volume goes last and is added with weight 1: A_i(U z_i + mean_volume)
    pca_dim = pc_volumes.shape[0]
    all_volumes = np.vstack([pc_volumes, mean_volume])
    prepared_volumes = [simulator.prepare_volume_for_projection(vol, experiment_dataset, disc_type) for vol in all_volumes]
    noise_image_upsampled = simulator.upsampled_noise_image(noise_variance, experiment_dataset.image_shape)
    image_shape = tuple(experiment_dataset.image_shape)

    n_images_done = 0
    for k in range(0, int(np.ceil(n_images / batch_size))):
        batch_st = int(k * batch_size)
        batch_end = int(np.min([(k + 1) * batch_size, n_images]))
        indices = np.arange(batch_st, batch_end)

        rotation_matrices, translations, ctf_params = experiment_dataset.metadata.get_batch(indices)
        translations = np.zeros_like(translations) if pad_before_translate else translations

        # Sum the CTF-free projections of all volumes for this batch
        images_batch = np.zeros([indices.size, *image_shape], dtype=np.complex64)
        for vol_idx, volume in enumerate(prepared_volumes):
            projections = np.array(
                simulator.project_image_batch(
                    experiment_dataset,
                    volume,
                    rotation_matrices,
                    translations,
                    ctf_params,
                    disc_type,
                    skip_ctf=True,
                    premultiplied_ctf=premultiplied_ctf,
                ).reshape([-1, *image_shape])
            )
            if vol_idx < pca_dim:
                images_batch += image_assignments[indices, vol_idx, None, None] * projections
            else:
                images_batch += projections

        output_array[indices] = simulator.apply_ctf_and_noise_upsampled(
            experiment_dataset,
            images_batch,
            ctf_params,
            translations,
            indices,
            noise_image_upsampled,
            noise_subkeys,
            noise_rng_batch_size,
            batch_st,
            batch_end,
            n_images,
            per_image_noise_scale,
            per_image_contrast,
            premultiplied_ctf,
            pad_before_translate,
        )

        n_images_done += indices.size
        logger.info("Batch %s: Generated %s images so far", k, n_images_done)

    logger.info("Discretizing with: %s", disc_type)
    logger.info("Done generating data")

    if mrc_file is not None:
        return mrc_file
    else:
        return output_array
