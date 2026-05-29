import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import os

from recovar.heterogeneity import latent_density as ld
from recovar.output import output as o


XLA_PYTHON_CLIENT_PREALLOCATE=False

@jax.jit
def compute_grad(weights, zs, cov_zs):
    #model = likelihood @ weights
    #grad = jnp.mean(likelihood/model[:, jnp.newaxis], axis=0)
    return None


@jax.jit
def update_weights(weights, grad):
    """
    Updates weights according to multiplicative gradient algorithm.
    NOTE: this update is positive and sums to 1 without normalization.

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    grad : jax.Array
        gradient of weights, same shape as weights

    Returns
    -------
    updated weights: jax.Array  
    """
    return weights*grad


@jax.jit
def scaled_gap(grad, weights, scale):
    """
    Find maximum index of gradient vec, only at nonzero indices of weights, and rescale.
    This gap is a proxy for convergence, common for convex objectives.

    Parameters
    ----------
    grad : jax.Array
        gradient of weights, same shape as weights
    weights : jax.Array
        weights of the nodes
    scale : float
        scaling factor, so that the gap at initial iterate is 1. 

    Returns
    -------
    scaled gap: float
    """
    grad = jnp.where(weights > 0, grad, 0)
    return (jnp.amax(grad) - 1) / scale


def online_multiplicative_gradient(
    recovar_result_dir,
    zdim,
    tol=1e-2,
    max_iterations=10000,
):

    path =  os.path.abspath(recovar_result_dir + '/')
    po = o.PipelineOutput(path)

    zs = po.get_embedding_component("latent_coords_noreg", 2)
    cov_zs= po.get_embedding_component("latent_precision_noreg", 2)

    ## Throwing away unstable covariances
    cov_zs_norm = jnp.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm > jnp.percentile(cov_zs_norm, percentile_reject=10)
    zs = zs[good_zs][:,:zdim]
    cov_zs = cov_zs[good_zs][:,:zdim, :zdim]

    ## Making a grid
    latent_space_bounds = ld.compute_latent_space_bounds(zs, percentile=1)
    if zdim == 1:
        num_points_per_dim = 500
    elif zdim == 2:
        num_points_per_dim = 200
    elif zdim > 2:
        num_points_per_dim = 50

    grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points_per_dim)
    nodes = grids_flat.reshape(num_points_per_dim**zdim, grids_flat.shape[-1])

    num_nodes = num_points_per_dim**zdim

    # Initialize weights
    weights = (1/num_nodes)*jnp.ones(num_nodes)

    #TODO: per line of the gradient, normalize the log likelihood
    # Convert log likelihood to likelihood via "soft-max"-ish operation
    #likelihood = normalize_log_likeli_to_likeli(log_likelihood)
  
    # Initialize scaling for gap stopping criteria
    gap_scale = scaled_gap(compute_grad(zs, cov_zs, nodes), weights, scale=1.0)

    reached_gap = False
    for k in range(max_iterations):
        # Update grad
        grad = compute_grad(zs, cov_zs, nodes)

        # Check stopping criterions
        gap = scaled_gap(grad, weights, gap_scale)

        # Check current gap against tolerance
        if not reached_gap and gap < tol:
            print(f"reached gap tolerance, at idx: {k}")
            print(f"gap: {gap}")
            reached_gap = True
        
        # Check if stopping criteria met
        if reached_gap:
            print(f"exiting! At iteration: {k}")
            break

        # Update weights
        weights = update_weights(weights, grad)

    return weights

def main():
    # Set up RNG keys
    seed_train_test = 1

    # Set up directories (not needed here but can be used if wanting saved figs)
    main_dir = "."
    fig_dir = f"{main_dir}/figures/hsp90"
    data_dir = f"{main_dir}/data/"
    #os.makedirs(fig_dir, exist_ok=True)
    #os.makedirs(data_dir, exist_ok=True)

    # Set up pretty plots 
    plt.style.use("my_style.mplstyle") # Use stylefile defined
    plt.style.use("seaborn-v0_8-colorblind") # Use colorscheme from colorblind seaborn

    # Load grid
    #grid = jnp.load("/mnt/home/levans/ceph/archive/cryo_reweighting_examples/latent_example/zs_grid_new.npy")
    #print(grid.shape)

    # Load likelihood matrix
    log_likelihood = -1*jnp.load("/mnt/home/levans/ceph/archive/cryo_reweighting_examples/latent_example/log_likelihood_zs_grid_zdim_2.npy").astype('float16')
    num_data, num_nodes = jnp.shape(log_likelihood)
    print(log_likelihood.shape)

    ##true_weights = jnp.load(f"{data_dir}/hsp90_true_weights.npy")
    #weights, info = opt.multiplicative_gradient(log_likelihood, 
    #                                                max_iterations=10000, 
    #                                                weights_frequency=1,
    #                                                tol = 1e-2, 
    #                                                verbose=True, 
    #                                                train_test=False,
    #                                                diagnostic=False)
    #this code will plot `weights` returned above as the max iteration weights in the plots: 
    weights, info = online_multiplicative_gradient(recovar_result_dir,
                                                   zdim=2,
    )
    #this code will plot `weights` returned above as the max iteration weights in the plots: 

    
    
    jnp.savez("weights.npy", weights) 
    plt.imshow(weights.reshape(200,200).T, cmap="magma")
    plt.savefig("plot_fig.png",dpi=300) 
    plt.show()


if __name__ == "__main__":
    main()
