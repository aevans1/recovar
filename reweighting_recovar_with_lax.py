import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= "false"

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from recovar.heterogeneity import latent_density as ld
from recovar.output import output as o
from recovar import utils

# NOTE: this multipyling by exp(c) could be unstable, need to refactor below to be in log space, and also avoid taking log pi, by using some weighted log sum exp functions
# The exp(c) needs to be there right now because, on a batch of grid points, this constant won't cancel with the denominator (computed over the full set of grid points)
@eqx.filter_jit
def custom_exp_normalize(arr, axis): 
   c = jnp.amax(arr, axis=axis, keepdims=True)
   return jnp.exp(arr - c)*jnp.exp(c)


@jax.jit
def grad_accum(grad, val, c):
    return grad + c*val


@eqx.filter_jit
def compute_denom_and_grad_image_batch(weights, zs_batch, cov_zs_batch, zs_grid, batch_size=1000):
    num_nodes = zs_grid.shape[0]
    num_full_chunks = num_nodes // batch_size
    remainder = num_nodes % batch_size

    ## First pass through zs_grid batches: computing denominator of the gradient summation, for the image batch
    # Computing denominator for image batch (zs_batch, cov_zs_batch), over all full batches of nodes (zs_grid)
    def denom_body_fn(denom_batch_accum, idx):
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx*batch_size, batch_size, axis=0)
        weights_chunk = jax.lax.dynamic_slice_in_dim(weights, idx*batch_size, batch_size, axis=0)
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
        denom_batch_accum += exp_norm @ weights_chunk
        return denom_batch_accum, None

    denominator_init = jnp.zeros(zs_batch.shape[0])
    denominator, _ = jax.lax.scan(denom_body_fn, denominator_init, jnp.arange(num_full_chunks))

    # Remainder calc: computing denominator for image batch (zs_batch, cov_zs_batch), over last batch of nodes (zs_grid) if any leftover
    if remainder > 0:
        grid_chunk = zs_grid[num_full_chunks*batch_size:]
        weights_chunk = weights[num_full_chunks*batch_size:]
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        exp_norm_remainder = custom_exp_normalize(log_likelihood_batch, axis=1)
        denominator += exp_norm_remainder @ weights_chunk

    ## Second pass through zs_grid batches: computing the gradient summation, for the image batch
    # Computing gradient for image batch (zs_batch, cov_zs_batch), over all full batches of nodes (zs_grid)   
    def grad_body_fn(grad_batch_accum, idx):
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx*batch_size, batch_size, axis=0)
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
        val = jnp.sum(exp_norm / denominator[:, None], axis=0)
        grad_batch_accum = jax.lax.dynamic_update_slice(grad_batch_accum, val, (idx*batch_size,))
        return grad_batch_accum, None

    grad_batch_init = jnp.zeros(zs_grid.shape[0]) 
    grad_batch, _ = jax.lax.scan(grad_body_fn, grad_batch_init, (jnp.arange(num_full_chunks)))

    # Remainder calc: computing grad for image batch (zs_batch), over last batch of confs (zs_grid) if any leftover
    if remainder > 0:
        val = jnp.sum(exp_norm_remainder / denominator[:, None], axis=0)
        grad_batch = grad_batch.at[num_full_chunks*batch_size:].set(val)
    
    return grad_batch


def compute_grad(weights, zs, cov_zs, zs_grid, batch_size_zs=1000, batch_size_nodes=1000):
    """Gradient computation, batched over zs (embedded images), and batched over zs_grid (latent volumes/confs). 
    For the batching, a log likelihood matrix cannot be pre-computed, its pre-computed at iteration.
    This is a naive first try at this, maybe there is a way of caching some of these to not re-use on subsequent evaluations....
    This is a batched version of the following code, with its own docstring: 

    ######################################################################### 
    ----
    Evaluate the gradient of the log-likelihood of the data given the weights.

    This computes the "probabilistic model" for the data prob density with weights w
    - sum_j p(y_i |x_j) w_j 
    And then computes the gradient of (1/num_data)*sum_i log (sum_j p(y_i|x_j) w_j):
    - (1/num_data)*sum_i ((p(y_i|x_j) / sum_k p(y_i | x_k) w_j))

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 

    Returns
    -------
    gradient of log marginal likelihood: jax.Array
    
    model = likelihood @ weights
    grad = jnp.mean(likelihood/model[:, jnp.newaxis], axis=0)
    return grad
    ######################################################################### 
    """

    # Batching through zs, getting full batches and remainder batch 
    num_data = zs.shape[0]
    num_full_chunks = num_data // batch_size_zs
    remainder = num_data % batch_size_zs

    grad = jnp.zeros(zs_grid.shape[0])
    for i in range(num_full_chunks):
        zs_chunk = zs[i*batch_size_zs:(i + 1)*batch_size_zs, :]
        cov_zs_chunk =  cov_zs[i*batch_size_zs:(i + 1)*batch_size_zs, :]
        val = compute_denom_and_grad_image_batch(weights, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size_nodes)
        grad = grad_accum(grad, val, 1/zs.shape[0])
   
    ## Computing gradient over last batch of images if any leftover
    if remainder > 0:
        zs_chunk = zs[num_full_chunks*batch_size_zs:]
        cov_zs_chunk = cov_zs[num_full_chunks*batch_size_zs:]
        val = compute_denom_and_grad_image_batch(weights, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size_nodes)
        grad = grad_accum(grad, val, 1/zs.shape[0])
   
    return grad

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
    Originally, find maximum index of gradient vec, only at nonzero indices of weights, and rescale.
    Instead, using a more stable criteria here (first is commented out).
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
    
    #NOTE: the max grad seems very unstable for these batched compuations, more reason to use a different criteria, like the weighted square norm below
    #grad = jnp.where(weights > 0, grad, 0)
    #return (jnp.amax(grad) - 1) / scale
    return jnp.sum(weights*(grad - 1)**2) / scale


def online_multiplicative_gradient(
    recovar_result_dir,
    zdim,
    tol=1e-2,
    max_iterations=10000,
    batch_size_zs=1000,
    batch_size_nodes=1000
):

    path =  os.path.abspath(recovar_result_dir + '/')
    po = o.PipelineOutput(path)

    zs = po.get_embedding_component("latent_coords_noreg", zdim)
    cov_zs= po.get_embedding_component("latent_precision_noreg", zdim)

    ## Throwing away unstable covariances
    cov_zs_norm = jnp.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm > jnp.percentile(cov_zs_norm, q=10)
    zs = zs[good_zs][:,:zdim]
    cov_zs = cov_zs[good_zs][:,:zdim, :zdim]

    ## Making a grid
    latent_space_bounds = ld.compute_latent_space_bounds(zs, percentile=1)
    if zdim == 1:
        num_points_per_dim = 500
    elif zdim == 2:
        num_points_per_dim = 200
    elif zdim > 2:
        num_points_per_dim = 20
        num_points_per_dim = 50 # 50 won't run fast enough :(
    grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points_per_dim)
    nodes = grids_flat.reshape(num_points_per_dim**zdim, grids_flat.shape[-1])

    num_nodes = num_points_per_dim**zdim

    # Initialize weights
    weights = (1/num_nodes)*jnp.ones(num_nodes)

    # Initialize scaling for gap stopping criteria
    gap_scale = scaled_gap(compute_grad(weights, zs, cov_zs, nodes, batch_size_zs=batch_size_zs, batch_size_nodes=batch_size_nodes), weights, scale=1.0)
    reached_gap = False

    for k in range(max_iterations):
        # Update grad
        grad = compute_grad(weights, zs, cov_zs, nodes, batch_size_zs=batch_size_zs, batch_size_nodes=batch_size_nodes)
        print(k)
        # Check stopping criterions
        gap = scaled_gap(grad, weights, gap_scale)
        print(f"gap: {gap}")
        print(f"sum of weights: {jnp.sum(weights)}")
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


def plot_density(density, function=None, cmap="inferno"):
    from recovar.output.output import sum_over_other

    def half_slice_other(density, axes):
        axes = [i for i in range(density.ndim) if i not in axes]
        axes = np.sort(axes)
        for i in range(len(axes) - 1, -1, -1):
            density = np.take(density, density.shape[0] // 2, axis=axes[i])
        return density

    function = half_slice_other if function == "slice" else function
    function = sum_over_other if function is None else function

    plt.rcParams.update({})
    font = {"weight": "bold", "size": 22}
    import matplotlib

    matplotlib.rc("font", **font)

    n_plots = 2
    n_cols = density.ndim + 1 if density.ndim < 2 else density.ndim
    fig, axs = plt.subplots(n_plots, n_cols, figsize=(n_cols * 5, n_plots * 5))
    print(axs.shape)
    global is_first
    is_first = True

    def plot_dens(density, title, n_plot, first=False):
        density = np.asarray(density)
        global is_first
        if density.ndim == 1:
            axs[n_plot, 0].plot(density)
            axs[n_plot, 0].set_title(title)
            axs[n_plot, 0].set_xticklabels([])
            axs[n_plot, 0].set_yticklabels([])
            if is_first:
                axs[n_plot, 0].set_title(f"PC x={0}")
            axs[n_plot, 0].set_ylabel(title)
            return

        for k in range(1, density.ndim):
            if k == 1:
                axs[n_plot, k - 1].set_ylabel(title)
            axs[n_plot, k - 1].set_xticklabels([])
            axs[n_plot, k - 1].set_yticklabels([])

            to_plot = function(density, [0, k])
            axs[n_plot, k - 1].imshow(to_plot.T, cmap=cmap)
            if is_first:
                axs[n_plot, k - 1].set_title(f"PC x={0}, y={k}")

        if density.ndim > 2:
            to_plot = function(density, [1, 2])
            axs[n_plot, k].imshow(to_plot.T, cmap=cmap)
            axs[n_plot, k].set_xticklabels([])
            axs[n_plot, k].set_yticklabels([])
            if is_first:
                axs[n_plot, k].set_title(f"PC x={1}, y={2}")
        is_first = False


    plot_dens(density, "MG density", 0)

    plt.subplots_adjust(wspace=0, hspace=0)



def main():
    # Set up RNG keys
    seed_train_test = 1

    # Set up directories (not needed here but can be used if wanting saved figs)
    main_dir = "."
    fig_dir = f"{main_dir}/figures/hsp90"
    data_dir = f"{main_dir}/data/"

    # Set dir with all zs, cov_zs
    recovar_result_dir="/mnt/home/levans/ceph/recovar_testing/bad_histogram_igg/_given_mask_with_correct_contrast"


    # Set params for algorithm
    zdim=2
    batch_size_zs = 25000
    batch_size_nodes = 25000

    weights = online_multiplicative_gradient(recovar_result_dir,
                                             zdim=zdim,
                                             tol=1e-8,
                                             max_iterations=10,
                                             batch_size_zs=batch_size_zs,
                                             batch_size_nodes=batch_size_nodes
    )
    
    #jnp.save(f"weights_online_zdim_{zdim}.npy", weights) 
    
    if zdim == 2: 
        plt.imshow(weights.reshape(200,200).T, cmap="magma")
        plt.savefig(f"plot_fig_weights_online_{zdim}.png",dpi=300) 

    if zdim == 4: 
        plot_density(weights.reshape(20,20,20,20)) 
        plt.savefig("plot_density_recovar_style.png",dpi=300) 
        plt.savefig("plot_density_recovar_style_4D.png",dpi=300) 
    plt.show()

if __name__ == "__main__":
    main()


# This is an experiment in speeding up computations by caching some things, but the speedup doesn't seem worth it
#@eqx.filter_jit
#def compute_denom_and_grad_image_batch_cache_exp_norm(weights, zs_batch, cov_zs_batch, zs_grid, batch_size=1000):
#    num_nodes = zs_grid.shape[0]
#    num_full_chunks = num_nodes // batch_size
#    remainder = num_nodes % batch_size
#
#    # Computing denominator for image batch (zs_batch), over all full batches of confs (zs_grid)
#    def denom_body_fn(denom_batch_accum, idx):
#        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx*batch_size, batch_size, axis=0)
#        weights_chunk = jax.lax.dynamic_slice_in_dim(weights, idx*batch_size, batch_size, axis=0)
#        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
#        exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
#        denom_batch_accum += exp_norm @ weights_chunk
#        return denom_batch_accum , exp_norm
#
#    denominator_init = jnp.zeros(zs_batch.shape[0])
#    denominator, exp_norm_chunks = jax.lax.scan(denom_body_fn, denominator_init, jnp.arange(num_full_chunks))
#
#    # Computing denominator for image batch (zs_batch), over last batch of confs (zs_grid) if any leftover
#    if remainder > 0:
#        grid_chunk = zs_grid[num_full_chunks*batch_size:]
#        weights_chunk = weights[num_full_chunks*batch_size:]
#        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
#        exp_norm_remainder = custom_exp_normalize(log_likelihood_batch, axis=1)
#        denominator += exp_norm_remainder @ weights_chunk
#
#
#    # Computing gradient for image batch (zs_batch), over all full batches of confs (zs_grid)   
#    def grad_body_fn(grad_batch_accum, carry):
#        idx, exp_norm = carry
#        val = jnp.sum(exp_norm / denominator[:, None], axis=0)
#        grad_batch_accum = jax.lax.dynamic_update_slice(grad_batch_accum, val, (idx*batch_size,))
#        return grad_batch_accum, None
#
#    grad_batch_init = jnp.zeros(zs_grid.shape[0]) 
#    grad_batch, _ = jax.lax.scan(grad_body_fn, grad_batch_init, (jnp.arange(num_full_chunks), exp_norm_chunks))
#
#    # Computing grad for image batch (zs_batch), over last batch of confs (zs_grid) if any leftover
#    if remainder > 0:
#        val = jnp.sum(exp_norm_remainder / denominator[:, None], axis=0)
#        grad_batch = grad_batch.at[num_full_chunks*batch_size:].set(val)
#    
#    return grad_batch

