
XLA_PYTHON_CLIENT_PREALLOCATE=False

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

import os

from recovar.heterogeneity import latent_density as ld
from recovar.output import output as o
from recovar import utils

### Custom exp stuff
@eqx.filter_jit
def custom_exp_normalize(arr, axis): 
   c = jnp.amax(arr, axis=axis, keepdims=True)
   return jnp.exp(arr - c)*jnp.exp(c)

#### Experimental, batching images AND confs
def compute_denom_image_batch(weights, zs_batch, cov_zs_batch, zs_grid, batch_size=1000):
    """jit compiling last isolated part of the gradient"""
    num_nodes = zs_grid.shape[0]
    num_full_chunks = num_nodes // batch_size
    remainder = num_nodes % batch_size
    def body_fn(carry, idx):
        denominator = carry
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx*batch_size, batch_size, axis=0)
        weights_chunk = jax.lax.dynamic_slice_in_dim(weights, idx*batch_size, batch_size, axis=0)
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        return carry + custom_exp_normalize(log_likelihood_batch, axis=1) @ weights_chunk, None

    denominator_init = jnp.zeros(zs_batch.shape[0])
    denominator, _ = jax.lax.scan(body_fn, denominator_init, jnp.arange(num_full_chunks))

    # Add remainder batch 
    def compute_remainder_batch(denominator):
        grid_chunk = zs_grid[num_full_chunks*batch_size:]
        weights_chunk = weights[num_full_chunks*batch_size:]
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        denominator += custom_exp_normalize(log_likelihood_batch, axis=1) @ weights_chunk
        return denominator

    # Add conditional here because we can't jit compile through a usual conditioning
    print(remainder)
    denominator = jax.lax.cond(remainder > 0, compute_remainder_batch, lambda g: g, denominator) 
    return denominator
 

def _compute_grad_batch_alt(denominator, zs_batch, cov_zs_batch, zs_grid, batch_size=1000):
    """gradient computation, in batches of images AND latents"""
    num_nodes = zs_grid.shape[0]
    num_full_chunks = num_nodes // batch_size
    remainder = num_nodes % batch_size

    def body_fn(grad_batch_accum, idx):
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx*batch_size, batch_size, axis=0)
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        val = jnp.sum(custom_exp_normalize(log_likelihood_batch, axis=1) / denominator[:, None], axis=0)
        grad_batch_accum = jax.lax.dynamic_update_slice(grad_batch_accum, val, (idx*batch_size,))
        return grad_batch_accum, None

    grad_batch_init = jnp.zeros(zs_grid.shape[0]) 
    grad_batch, _ = jax.lax.scan(body_fn, grad_batch_init, jnp.arange(num_full_chunks))

    # Add remainder batch 
    def compute_remainder_batch(grad_batch):
        grid_chunk = zs_grid[num_full_chunks*batch_size:]
        log_likelihood_batch = -1*ld.compute_latent_log_likelihood(grid_chunk, zs_batch, cov_zs_batch, batch_size=batch_size)
        val = jnp.sum(custom_exp_normalize(log_likelihood_batch, axis=1) / denominator[:, None], axis=0)
        grad_batch = grad_batch.at[num_full_chunks*batch_size:].set(val)
        return grad_batch

    # Add conditional here because we can't jit compile through a usual conditioning
    grad_batch = jax.lax.cond(remainder > 0, compute_remainder_batch, lambda g: g, grad_batch) 
    return grad_batch 
    
def compute_grad(weights, zs, cov_zs, zs_grid, batch_size=1000):
    # Batching through zs, getting full batches and remainder batch 
    num_data = zs.shape[0]
    num_full_chunks = num_data // batch_size
    remainder = num_data % batch_size
    
    # Compute grad for batches of full size 
    def body_fn(accum, idx):
        grad = accum 
        zs_chunk = jax.lax.dynamic_slice_in_dim(zs, idx*batch_size, batch_size, axis=0)
        cov_zs_chunk = jax.lax.dynamic_slice_in_dim(cov_zs, idx*batch_size, batch_size, axis=0)
        denominator = compute_denom_image_batch(weights, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size) 
        val = _compute_grad_batch_alt(denominator, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size)
        return grad + (1 / zs.shape[0])*val, None

    grad_init = jnp.zeros(zs_grid.shape[0])
    grad, _ = jax.lax.scan(body_fn, grad_init, jnp.arange(num_full_chunks))
    
    # Add grad for remainder batch 
    def compute_remainder_batch(grad):
        zs_chunk = zs[num_full_chunks*batch_size:]
        cov_zs_chunk = cov_zs[num_full_chunks*batch_size:]
        denominator = compute_denom_image_batch(weights, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size) 
        val = _compute_grad_batch_alt(denominator, zs_chunk, cov_zs_chunk, zs_grid, batch_size=batch_size)
        return grad + (1 / zs.shape[0])*val

    # Add conditional here because we can't jit compile through a usual conditioning
    grad = jax.lax.cond(remainder > 0, compute_remainder_batch, lambda g: g, grad) 
    
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
    #grad = jnp.where(weights > 0, grad, 0)
    #return (jnp.amax(grad) - 1) / scale
    return jnp.sum(weights*(grad - 1)**2)


def online_multiplicative_gradient(
    recovar_result_dir,
    zdim,
    tol=1e-2,
    max_iterations=10000,
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
        num_points_per_dim = 50

    grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points_per_dim)
    nodes = grids_flat.reshape(num_points_per_dim**zdim, grids_flat.shape[-1])

    num_nodes = num_points_per_dim**zdim

    # Initialize weights
    weights = (1/num_nodes)*jnp.ones(num_nodes)

    # Getting batch size from GPU
    batch_size_zs = 25000 
    #batch_size_zs = utils.get_latent_density_batch_size(nodes, zs.shape[-1], utils.get_gpu_memory_total())
    #print(f"batch size zs: {batch_size_zs}")

    #TODO: per line of the gradient, normalize the log likelihood
    # Convert log likelihood to likelihood via "soft-max"-ish operation
    #likelihood = normalize_log_likeli_to_likeli(log_likelihood)
  
    # Initialize scaling for gap stopping criteria
    gap_scale = scaled_gap(compute_grad(weights, zs, cov_zs, nodes, batch_size=batch_size_zs), weights, scale=1.0)
    reached_gap = False

    for k in range(max_iterations):
        # Update grad
        grad = compute_grad(weights, zs, cov_zs, nodes, batch_size=batch_size_zs)
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
    #os.makedirs(fig_dir, exist_ok=True)
    #os.makedirs(data_dir, exist_ok=True)

    # Set up pretty plots 
    #plt.style.use("my_style.mplstyle") # Use stylefile defined
    #plt.style.use("seaborn-v0_8-colorblind") # Use colorscheme from colorblind seaborn

    # Load grid
    #grid = jnp.load("/mnt/home/levans/ceph/archive/cryo_reweighting_examples/latent_example/zs_grid_new.npy")
    #print(grid.shape)

    ## Load likelihood matrix
    #log_likelihood = -1*jnp.load("/mnt/home/levans/ceph/archive/cryo_reweighting_examples/latent_example/log_likelihood_zs_grid_zdim_2.npy").astype('float16')
    #num_data, num_nodes = jnp.shape(log_likelihood)
    #print(log_likelihood.shape)

    #true_weights = jnp.load(f"{data_dir}/hsp90_true_weights.npy")
    #weights, info = opt.multiplicative_gradient(log_likelihood, 
    #                                                max_iterations=10000, 
    #                                                weights_frequency=1,
    #                                                tol = 1e-2, 
    #                                                verbose=True, 
    #                                                train_test=False,
    #                                                diagnostic=False)
    #this code will plot `weights` returned above as the max iteration weights in the plots: 
    recovar_result_dir="/mnt/home/levans/ceph/recovar_testing/bad_histogram_igg/_given_mask_with_correct_contrast"
    
    zdim=4
    ##weights = online_multiplicative_gradient(recovar_result_dir,
    ##                                               zdim=2
    ##)
    weights = online_multiplicative_gradient(recovar_result_dir,
                                             zdim=zdim,
                                             tol=1e-8,
                                             max_iterations=50,
    )
    
    #jnp.save(f"weights_online_zdim_{zdim}.npy", weights) 
    
    if zdim == 2: 
        plt.imshow(weights.reshape(200,200).T, cmap="magma")
        plt.savefig(f"plot_fig_weights_online_{zdim}.png",dpi=300) 

    if zdim == 4: 
        plot_density(weights.reshape(50,50,50,50)) 
        plt.savefig("plot_density_recovar_style.png",dpi=300) 
        plt.savefig("plot_density_recovar_style_4D.png",dpi=300) 
    plt.show()


if __name__ == "__main__":
    main()
