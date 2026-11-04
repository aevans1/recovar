import argparse
import glob
import logging
import os

import numpy as np

from recovar import utils
from recovar.heterogeneity import embedding
from recovar.output import output
from recovar.output.output_paths import ResultPaths

logger = logging.getLogger(__name__)


# Flagging fields that will not be needed in the modified-in-place pipelines
STALE_PARAMS_FIELDS = ["variance_fsc", "column_fscs", "picked_frequencies"]
REGULARIZED_EMBEDDING_FIELDS = ["latent_coords", "latent_precision", "contrasts"]


def rescale_embedding(zs, precision, scale_vol):
    """Map's latent coordinates according to how volumes were scaled in the simulation"""
    return zs / scale_vol, precision * scale_vol**2


def to_nan_padded(arr, particles_ind_split):
    """keep things in right order with nans where needed"""
    sorted_indices = np.sort(np.concatenate(particles_ind_split))
    n_original = int(sorted_indices.max()) + 1
    out = np.full((n_original, *arr.shape[1:]), np.nan, dtype=arr.dtype)
    out[sorted_indices] = arr
    return out


def reembed_with_reference_pcs(result_dir, source_pipeline_dir, sim_dir, n_pcs=2, diagnostic_plots=False):
    paths = ResultPaths(result_dir)

    # get volume scaling and true noise from simulation
    sim_info = utils.pickle_load(os.path.join(sim_dir, "simulation_info.pkl"))
    scale_vol = sim_info["scale_vol"]

    # get other vol info
    params = utils.pickle_load(paths.params)
    volume_shape = params["volume_shape"]
    voxel_size = params["voxel_size"]

    # Load Mean, PCs and eigvals from the reference pipeline
    po_reference = output.PipelineOutput(source_pipeline_dir)
    new_eigen_volumes_ft = po_reference.get_u()
    new_mean_volume_ft = po_reference.get("mean")
    new_s = po_reference.get("s")
    mask = utils.load_mrc(paths.mask_volume)

    for f in glob.glob(os.path.join(paths.volumes_dir, "eigen_pos*.mrc")):
        os.remove(f)

    output.save_covar_output_volumes(
        paths.output_dir,
        new_mean_volume_ft,
        new_eigen_volumes_ft.T,
        {"rescaled": new_s},
        mask,
        volume_shape,
        us_to_save=new_eigen_volumes_ft.shape[-1],
        voxel_size=voxel_size,
    )

    # replacing eigvals
    params["s"] = new_s

    # clearing covar outputs from old pcs, logging it
    for stale_key in STALE_PARAMS_FIELDS:
        params[stale_key] = None
    if os.path.exists(paths.covariance_cols):
        os.remove(paths.covariance_cols)
    params["pipeline_eigvecs_replaced_from"] = source_pipeline_dir
    params["regularized_embeddings_removed"] = True
    utils.pickle_dump(params, paths.params)
    logger.info("Replaced mean.mrc, eigen_pos*.mrc, variance*.mrc and params['s'] in %s", result_dir)

    # Re-embed with no prior and the true simulator noise, against scale_vol * volumes, like in sim
    po = output.PipelineOutput(result_dir)
    u = po.get_u(n_pcs=n_pcs)
    s = po.get("s")[:n_pcs] * 0 + np.inf
    dataset = po.get("dataset")
    dataset.set_noise(sim_info["noise_variance"])

    zs, precision, contrasts, _ = embedding.get_per_image_embedding(
        scale_vol * po.get("mean"), u.T, s, n_pcs, dataset, po.get("volume_mask"), utils.get_gpu_memory_total()
    )
    zs, precision = rescale_embedding(zs, precision, scale_vol)

    particles_ind_split = po.get("particles_halfsets")
    zdim_dir = paths.embedding_zdim_dir(zdim=n_pcs)
    os.makedirs(zdim_dir, exist_ok=True)
    np.save(os.path.join(zdim_dir, "latent_coords_noreg.npy"), to_nan_padded(zs, particles_ind_split))
    np.save(os.path.join(zdim_dir, "latent_precision_noreg.npy"), to_nan_padded(precision, particles_ind_split))
    np.save(os.path.join(zdim_dir, "contrasts_noreg.npy"), to_nan_padded(contrasts, particles_ind_split))

    # TODO: figure out what to do with regularized embeddings?
    # Removing reg embeddings: they are not needed for now, haven't been adjusted
    for field in REGULARIZED_EMBEDDING_FIELDS:
        path = os.path.join(zdim_dir, f"{field}.npy")
        if os.path.exists(path):
            os.remove(path)
    logger.info("Saved unregularized embeddings for zdim=%s in %s; removed regularized ones", n_pcs, zdim_dir)

    if diagnostic_plots:
        _plot_true_vs_reembedded(sim_info["image_assignment"], zs, result_dir)

    return zs, precision, contrasts


def _plot_true_vs_reembedded(true_image_assignments, zs, output_dir):
    import matplotlib.pyplot as plt

    from recovar.heterogeneity import latent_density as ld

    true_latent_bounds = ld.compute_latent_space_bounds(true_image_assignments, percentile=1)
    noisy_latent_bounds = ld.compute_latent_space_bounds(zs, percentile=1)
    for bounds, name in [(true_latent_bounds, "true"), (noisy_latent_bounds, "noisy")]:
        plt.figure()
        plt.scatter(true_image_assignments[:, 0], true_image_assignments[:, 1], s=0.1, alpha=0.5)
        plt.scatter(zs[:, 0], zs[:, 1], s=0.1, alpha=0.5)
        plt.xlim(bounds[0, 0], bounds[0, 1])
        plt.ylim(bounds[1, 0], bounds[1, 1])
        plt.savefig(os.path.join(output_dir, f"simulated_true_scatter_on_{name}_bounds.png"), dpi=300)
        plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", help="Pipeline result on simulated data (to be modified in place)")
    parser.add_argument(
        "--source-pipeline-dir",
        required=True,
        help="Reference pipeline whose mean, PCs, eigvals used to replace result_dir's",
    )
    parser.add_argument(
        "--sim-dir",
        required=True,
        help="Simulation folder with simulation_info.pkl (provides scale_vol and the true noise variance)",
    )
    parser.add_argument("--n-pcs", type=int, default=2, help="Number of reference PCs to embed with (default 2)")
    parser.add_argument(
        "--diagnostic-plots",
        action="store_true",
        help="Save scatter plots of the true latents against the re-embedded ones in result_dir",
    )
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    reembed_with_reference_pcs(
        result_dir=args.result_dir,
        source_pipeline_dir=args.source_pipeline_dir,
        sim_dir=args.sim_dir,
        n_pcs=args.n_pcs,
        diagnostic_plots=args.diagnostic_plots,
    )


if __name__ == "__main__":
    main()
