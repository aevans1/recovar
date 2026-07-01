import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from recovar.heterogeneity import latent_density as ld
from recovar import utils

logger = logging.getLogger(__name__)

### Methods for "regular" multiplicative gradient #########################
@jax.jit
def normalize_log_likeli_to_likeli(log_likelihood):
    """
    Subtracts the largest entry from each row of the log likelihood.
    This is for stability, before transforming to likelihood (like in a soft-max)
    The gradient is invariant to row scaling of likelihood, so this is valid.
    With this normalizing, we avoid working in log space for the grad and loss.
    (only applies)

    Parameters
    ----------
    log_likelihood : jax.Array
        log_likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes)

    Returns
    -------
    likelihood: jax.Array
    """
    log_likelihood -= jnp.amax(log_likelihood, axis=1)[:, None]
    likelihood = jnp.exp(log_likelihood)
    return likelihood


@jax.jit
def compute_grad_and_loss(weights, likelihood):
    """
    This computes the "probabilistic model" for the data prob density with weights w
    And then computes log marginal likelihood, it's gradient of it's log with respect to the w's,
    and the negative log marginal likelihood (the loss)

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes)

    Returns
    -------
    (grad) gradient of log marginal likelihood: jax.Array
    (loss) -1* log marginal likelihood: float
    """

    model = likelihood @ weights
    loss = -jnp.mean(jnp.log(model))
    grad = jnp.mean(likelihood / model[:, jnp.newaxis], axis=0)
    return grad, loss


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
    return weights * grad


@jax.jit
def scaled_gap(grad, weights, scale):
    """
    Originally, find maximum index of gradient vec, only at nonzero indices of weights, and rescale.
    Instead, using a more stable criteria here (first is commented out).
    This gap is a proxy for convergence.

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

    # NOTE: the max grad seems very unstable for these batched computations, more reason to use a different criteria, like the weighted square norm below
    # grad = jnp.where(weights > 0, grad, 0)
    # return (jnp.amax(grad) - 1) / scale
    return jnp.sum(weights * (grad - 1) ** 2) / scale


### Methods for "online" multiplicative gradient ###############################
@eqx.filter_jit
def custom_exp_normalize(arr, axis):
    c = jnp.amax(arr, axis=axis, keepdims=True)
    return jnp.exp(arr - c)


@eqx.filter_jit
def grad_accum(grad, val, c):
    return grad + c * val

def compute_grad_and_loss_image_batch(weights, zs_batch, cov_zs_batch, det_cov_zs_batch, zs_grid, batch_size=1000):
    
    log_likelihood_batch = -1 * ld.compute_latent_log_likelihood_no_batch(
        zs_grid, zs_batch, cov_zs_batch, det_cov_zs_batch
    ).astype(jnp.float32)
    exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
    denominator = exp_norm @ weights
    grad_batch = jnp.sum(exp_norm / denominator[:, None], axis=0)
    return grad_batch, denominator.sum()


@eqx.filter_jit
def compute_online_grad_and_loss(weights, zs, cov_zs, det_cov_zs, zs_grid, batch_size_zs=1000):
    """Gradient and loss computation, batched over zs (embedded images).
    For the batching, a log likelihood matrix cannot be pre-computed, its pre-computed at iteration.
    This is a naive first try at this, maybe there is a way of caching some of these to not re-use on subsequent evaluations....
    This is a batched version of compute_grad()
    """
    ## Batching through zs, getting full batches and remainder batch
    num_data = zs.shape[0]
    num_full_chunks = num_data // batch_size_zs
    remainder = num_data % batch_size_zs

    ## Computing gradient over full batches
    def main_body_fn(grad_and_loss, idx):
        [grad, loss] = grad_and_loss
        zs_chunk = jax.lax.dynamic_slice_in_dim(zs, idx * batch_size_zs, batch_size_zs, axis=0)
        cov_zs_chunk = jax.lax.dynamic_slice_in_dim(cov_zs, idx * batch_size_zs, batch_size_zs, axis=0)
        det_cov_zs_chunk = jax.lax.dynamic_slice_in_dim(det_cov_zs, idx * batch_size_zs, batch_size_zs, axis=0)
        grad_val, loss_val = compute_grad_and_loss_image_batch(
            weights, zs_chunk, cov_zs_chunk, det_cov_zs_chunk, zs_grid)
        grad = grad_accum(grad, grad_val, 1 / zs.shape[0])
        loss += -(1 / num_data) * jnp.log(loss_val)
        return [grad, loss], None

    grad_and_loss_init = [jnp.zeros(zs_grid.shape[0], dtype=jnp.float32), jnp.float32(0.0)]
    grad_and_loss, _ = jax.lax.scan(main_body_fn, grad_and_loss_init, jnp.arange(num_full_chunks))
    [grad, loss]  = grad_and_loss
    
    ## Computing gradient over last batch of images if any leftover
    if remainder > 0:
        zs_chunk = zs[num_full_chunks * batch_size_zs :]
        cov_zs_chunk = cov_zs[num_full_chunks * batch_size_zs :]
        det_cov_zs_chunk = det_cov_zs[num_full_chunks * batch_size_zs :]
        grad_val, loss_val = compute_grad_and_loss_image_batch(
            weights, zs_chunk, cov_zs_chunk, det_cov_zs_chunk, zs_grid)
        grad = grad_accum(grad, grad_val, 1 / zs.shape[0])
        loss += -(1 / num_data) * jnp.log(loss_val)

    return grad, loss


# TODO: readjust this to return the loss per iteration and the gradient gap per iteration
def multiplicative_gradient(
    pipeline_output,
    z_dim_used=2,
    noreg=True,
    pca_dim=2,
    percentile_reject=10,
    num_points_per_dim=None,
    tol=1e-2,
    max_iterations=10000,
    online=True,
    weights_frequency=0,
    diagnostic=False,
    batch_size_zs=10000,
    ignore_cov_zs=False
):
    ## Load up latent image embeddings (zs) and uncertainties (cov_zs)
    coords_entry = "latent_coords_noreg" if noreg else "latent_coords"
    precision_entry = "latent_precision_noreg" if noreg else "latent_precision"

    zs = pipeline_output.get_embedding_component(coords_entry, z_dim_used).astype(jnp.float32)
    cov_zs = pipeline_output.get_embedding_component(precision_entry, z_dim_used).astype(jnp.float32)

    # selecting pca_dim subset of zs, cov_zs
    zs = zs[:, :pca_dim]
    cov_zs = cov_zs[:, :pca_dim, :pca_dim]

    ## Throwing away unstable covariances
    # TODO: should this outlier removing take place in z_dim_used, or pca_dim? For now, in pca_dim
    cov_zs_norm = jnp.linalg.norm(cov_zs, axis=(-1, -2), ord=2)
    good_zs = cov_zs_norm > jnp.percentile(cov_zs_norm, q=percentile_reject)
    zs = zs[good_zs]
    cov_zs = cov_zs[good_zs]
    det_cov_zs = ld.compute_log_det_cov(cov_zs).astype(jnp.float32)

    # NOTE: uncomment this for trying out the scalar diagonal approx
    if ignore_cov_zs:
        #cov_zs = jnp.tile(jnp.eye(2), (cov_zs.shape[0], 1, 1)).astype(jnp.float32)*jnp.mean(jnp.exp(det_cov_zs[:, None, None]))**(1/pca_dim)
        cov_zs = jnp.tile(jnp.mean(cov_zs, axis=0), (cov_zs.shape[0], 1, 1)).astype(jnp.float32)
        det_cov_zs = ld.compute_log_det_cov(cov_zs).astype(jnp.float32)

    ## Making a grid
    latent_space_bounds = ld.compute_latent_space_bounds(zs, percentile=1)
    if num_points_per_dim is None:
        if pca_dim == 1:
            num_points_per_dim = 500
        elif pca_dim == 2:
            num_points_per_dim = 200
        elif pca_dim > 2:
            num_points_per_dim = 50

    grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points_per_dim).astype(jnp.float32)
    nodes = grids_flat.reshape(num_points_per_dim**pca_dim, grids_flat.shape[-1])
    num_nodes = num_points_per_dim**pca_dim

    ## Initialize weights
    weights = (1 / num_nodes) * jnp.ones(num_nodes).astype(jnp.float32)


    ## Initialize info tracked
    info = {"losses": [], "gaps": [], "weights_all": [], "idx_weights": []}
    

    ## Initialize scaling for gap stopping criteria, and pre-compute log likelihood matrix if not online
    if online:
        ## Compute initial gap scale with online gradients
        logger.info("Using online mode for saving memory, recomputing likelihoods per gradient iteration")

        ## TODO: change to more precise control flow here with default options of None 
        if batch_size_zs == None:
            batch_size_zs = int(utils.get_latent_density_batch_size(nodes, zs.shape[-1], utils.get_gpu_memory_total()))
            logger.info("batch size of zs: %s", batch_size_zs)
            logger.info("number of nodes: %s", num_nodes)

        grad_init, loss_init = compute_online_grad_and_loss(
            weights, zs, cov_zs, det_cov_zs, nodes, batch_size_zs=batch_size_zs)
        gap_scale = scaled_gap(grad_init, weights, scale=1.0)
    else:
        ## Compute full likelihood matrix, re-use in non-online (full) gradient updates
        log_likelihood = -1 * ld.compute_latent_log_likelihood(nodes, zs, cov_zs).astype(jnp.float32)
        likelihood = normalize_log_likeli_to_likeli(log_likelihood)
        grad_init, loss_init  = compute_grad_and_loss(weights, likelihood)
        gap_scale = scaled_gap(grad_init, weights, scale=1.0)

    reached_gap = False
    for k in range(max_iterations):
        ## Save weights along the way if computing all the way to max iterations
        if weights_frequency > 0:
            if k==1 or (k % weights_frequency == 0 and k > 0):
                # NOTE: for now, setting weights frequency to start at iteration 1, not 0
                info["weights_all"].append(weights.reshape((num_points_per_dim,) * pca_dim))
                info["idx_weights"].append(k)

        ## Update grad and loss
        if online:
            grad, loss = compute_online_grad_and_loss(
                weights, zs, cov_zs, det_cov_zs, nodes, batch_size_zs=batch_size_zs)
        else:
            grad, loss = compute_grad_and_loss(weights, likelihood)
        info["losses"].append(loss)

        ## Check stopping criterions
        gap = scaled_gap(grad, weights, gap_scale)
        info["gaps"].append(gap)
 
        ## Logger info every 10 iterations
        if k % 10 == 0:
            logger.info(f" iteration {k}, gap: {gap}, loss: {loss}")

        ## Check current gap against tolerance
        if not reached_gap and gap < tol:
            info["gap_idx"] = k
            info["weights_gap"] = weights
            reached_gap = True
            logger.info(f"reached gap tolerance, at idx: {k}")
            logger.info(f"final gap: {gap}")
         
        ## Check if stopping criteria met
        if reached_gap and not diagnostic:
            logger.info(f"exiting! At iteration: {k}")
            break

        ## Update weights
        weights = update_weights(weights, grad)

    ## Return a grid of weights, and other info
    weights_final = weights.reshape((num_points_per_dim,) * pca_dim)
    if pca_dim==2:
        logger.info("temporary flipping of axis for compairsons in 2d...")
        for i in range(len(info["weights_all"])):
            info["weights_all"][i] = np.flip(info["weights_all"][i], axis=1)
    info["idx_final"] = k
    info["losses"] = jnp.stack(info["losses"])
    info["gaps"] = jnp.stack(info["gaps"])
    if weights_frequency > 0: 
        info["weights_all"] = jnp.stack(info["weights_all"])
        info["idx_weights"] = jnp.array(info["idx_weights"])
    if not reached_gap:
        logger.info("Terminated at max iters: ")
        logger.info("Returned weights & 'info[weights_gap']' are weights at max_iterations")
        info["weights_gap"] = weights_final
        info["idx_gap"] = k
    return weights_final, info


def plot_info(losses, gaps, plots_dir=None):
    num_iterations = len(losses)
    iterations = jnp.arange(num_iterations)

    plt.figure()
    plt.semilogy(iterations, losses)
    plt.xlabel("iterations")
    plt.ylabel("losses")
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(str(plots_dir / "losses.png"))

    plt.figure()
    plt.semilogy(iterations, gaps)
    plt.xlabel("iterations")
    plt.ylabel("gaps")
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(str(plots_dir / "gaps.png"))


# TODO: normalize colorbars? This isn't done in deconvolve_density, but maybe should be done for both?
def plot_density(densities, function=None, cmap="inferno", plots_dir=None, row_labels=None, cbar_normalize=False):
    from recovar.output.output import sum_over_other

    # Check if input is a array "list" of densities or a density
    if densities.ndim == 2:
        densities = np.expand_dims(densities, axis=0)
    
    # Find vmin, vmax for colorbars
    vmin = jnp.min(densities)
    vmax = jnp.max(densities)

    def half_slice_other(density, axes):
        other_axes = sorted([i for i in range(density.ndim) if i not in axes], reverse=True)
        for i in other_axes:
            density = np.take(density, density.shape[i] // 2, axis=i)
        return density

    function = half_slice_other if function == "slice" else function
    function = sum_over_other if function is None else function

    font = {"weight": "bold", "size": 22}
    import matplotlib

    matplotlib.rc("font", **font)

    n_rows = densities.shape[0]
    ndim = densities[0].ndim
    if ndim == 1:
        n_cols = 1
    else:
        n_pairs = ndim - 1
        n_extra = 1 if ndim > 2 else 0
        n_cols = n_pairs + n_extra

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    for row, density in enumerate(densities):
        if density.ndim == 1:
            axs[row, 0].plot(density)
            axs[row, 0].set_title("PC 0") if row == 0 else None
            axs[row, 0].set_xticklabels([])
            axs[row, 0].set_yticklabels([])
            if row_labels is not None:
                axs[row, 0].set_ylabel(row_labels[row])
            elif row == 0:
                axs[row, 0].set_ylabel("MG density")
        else:
            col = 0
            for k in range(1, density.ndim):
                to_plot = function(density, [0, k])
                if cbar_normalize:
                    axs[row, col].imshow(to_plot.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                else:
                    axs[row, col].imshow(to_plot.T, cmap=cmap, origin="lower")
                if row == 0:
                    axs[row, col].set_title(f"PC x=0, y={k}")
                axs[row, col].set_xticklabels([])
                axs[row, col].set_yticklabels([])
                if col == 0:
                    if row_labels is not None:
                        axs[row, col].set_ylabel(row_labels[row])
                    elif row == 0:
                        axs[row, col].set_ylabel("MG density")
                col += 1

            if density.ndim > 2:
                to_plot = function(density, [1, 2])
                if cbar_normalize:
                    axs[row, col].imshow(to_plot.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                else:
                    axs[row, col].imshow(to_plot.T, cmap=cmap, origin="lower")
                if row == 0:
                    axs[row, col].set_title("PC x=1, y=2")
                axs[row, col].set_xticklabels([])
                axs[row, col].set_yticklabels([])

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    if plots_dir is not None:
        if len(densities) == 0:
            plt.savefig(str(plots_dir / "final_density.png"), dpi=300)
        else:
            plt.savefig(str(plots_dir / "density_all.png"), dpi=300)
    return fig, axs
