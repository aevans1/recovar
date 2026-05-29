import argparse
import logging
import os

import jax.numpy as jnp

from recovar.heterogeneity import latent_density as ld
from recovar.output import output as o

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recovar_result_dir",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str
    )
    return parser.parse_args()

def compute_likelihoods_2d_grid(recovar_result_dir, output_dir, percentile_reject=10, percentile=1, reg=False):
    path =  os.path.abspath(recovar_result_dir + '/')
    po = o.PipelineOutput(path)

    ## Here, by default using nonregularized PC's, like the recovar paper, and throwing away unstable covariances
    if reg:
        zs = po.get_embedding_component("latent_coords", 2)
        cov_zs = po.get_embedding_component("latent_precision", 2)
    else:
        zs = po.get_embedding_component("latent_coords_noreg", 2)
        cov_zs= po.get_embedding_component("latent_precision_noreg", 2)

    ## Throwing away unstable covariances
    cov_zs_norm = jnp.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm > jnp.percentile(cov_zs_norm, percentile_reject)
    zs = zs[good_zs][:,:2]
    cov_zs = cov_zs[good_zs][:,:2, :2]

    ## Making a grid
    latent_space_bounds = ld.compute_latent_space_bounds(zs, percentile)
    num_points = 200
    grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points)
    zs_grid = grids_flat.reshape(num_points**2, grids_flat.shape[-1])

    ## compute likelihoods on the grid
    log_likelihood_zs_grid = ld.compute_latent_log_likelihood(zs_grid, zs, cov_zs)
    jnp.save(f"{output_dir}/log_likelihood_zs_grid_zdim_2.npy", log_likelihood_zs_grid)
    jnp.save(f"{output_dir}/zs_grid_zdim_2", zs_grid)

def compute_likelihoods_point_cloud(recovar_result_dir, output_dir, zdim=2, percentile_reject=10, reg=False):
    path =  os.path.abspath(recovar_result_dir + '/')
    po = o.PipelineOutput(path)

    ## Here, by default using nonregularized PC's, like the recovar paper, and throwing away unstable covariances
    if reg:
        zs = po.get_embedding_component("latent_coords", 2)
        cov_zs = po.get_embedding_component("latent_precision", 2)
    else:
        zs = po.get_embedding_component("latent_coords_noreg", 2)
        cov_zs= po.get_embedding_component("latent_precision_noreg", 2)

    ## Throwing away unstable covariances
    cov_zs_norm = jnp.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm > jnp.percentile(cov_zs_norm, percentile_reject)

    zs = zs[good_zs][:,:zdim]
    cov_zs = cov_zs[good_zs][:,:zdim, :zdim]

    ## Compute full likelihood matrix
    #log_likelihood_zs = ld.compute_latent_log_likelihood(zs, zs, cov_zs)

    ## Compute likelihood matrix for subset of z nodes, still all images, for easier computation
    #zs_subset = zs[::10, :]
    #log_likelihood_zs_subset = ld.compute_latent_log_likelihood(zs_subset, zs, cov_zs)

    ## Compute likelihood matrix for larger subset of latent nodes, AND subset of images, easier computation
    #zs_subset_larger = zs[::2, :]
    #cov_zs_subset_larger = cov_zs[::2, :, :]
    #log_likelihood_zs_subset_larger = ld.compute_latent_log_likelihood(zs_subset_larger, zs_subset_larger, cov_zs_subset_larger)

    ## Save likelihood matrices
    #np.save(f"{output_dir}/log_likelihood_zs_{zdim}.npy", log_likelihood_zs)
    #np.save(f"{output_dir}/log_likelihood_zs_subset_{zdim}.npy", log_likelihood_zs_subset)
    #np.save(f"{output_dir}/log_likelihood_zs_subset_larger_{zdim}.npy", log_likelihood_zs_subset_larger)

    ## Save the pc embeddings
    #np.save(f"{output_dir}/zs_zdim_{zdim}.npy", zs)
    #np.save(f"{output_dir}/zs_subset_{zdim}.npy", zs_subset)
    #np.save(f"{output_dir}/zs_subset_larger_{zdim}.npy", zs_subset_larger)

    ## Save covariances, for diagnostics
    #np.save(f"{output_dir}/cov_zs_{zdim}.npy", cov_zs)

def main():
    args = parse_args()
    compute_likelihoods_2d_grid(recovar_result_dir=args.recovar_result_dir,
                                output_dir=args.output_dir)

if __name__ == "__main__":
    main()
    