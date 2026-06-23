import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from recovar.heterogeneity import latent_density as ld

logger = logging.getLogger(__name__)


### Methods for "regular" multiplactive gradient #########################
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


### Methods for "online" multiplactive gradient ###############################


# TODO: this multipyling by exp(c) could be unstable, need to refactor below to be in log space, and also avoid taking log pi, by using some weighted log sum exp functions
# The exp(c) needs to be there right now because, on a batch of grid points, this constant won't cancel with the denominator (computed over the full set of grid points)
@eqx.filter_jit
def custom_exp_normalize(arr, axis):
    c = jnp.amax(arr, axis=axis, keepdims=True)
    return jnp.exp(arr - c) * jnp.exp(c)


@jax.jit
def grad_accum(grad, val, c):
    return grad + c * val


@eqx.filter_jit
def compute_grad_and_loss_image_batch(weights, zs_batch, cov_zs_batch, det_cov_zs_batch, zs_grid, batch_size=1000):
    num_nodes = zs_grid.shape[0]
    num_full_chunks = num_nodes // batch_size
    remainder = num_nodes % batch_size

    ## First pass through zs_grid batches: computing denominator of the gradient summation, for the image batch
    # Computing denominator for image batch (zs_batch, cov_zs_batch), over all full batches of nodes (zs_grid)
    def denom_body_fn(denom_batch_accum, idx):
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx * batch_size, batch_size, axis=0)
        weights_chunk = jax.lax.dynamic_slice_in_dim(weights, idx * batch_size, batch_size, axis=0)
        log_likelihood_batch = -1 * ld.compute_latent_log_likelihood_no_batch(
            grid_chunk, zs_batch, cov_zs_batch, det_cov_zs_batch
        ).astype(jnp.float32)
        exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
        return denom_batch_accum + exp_norm @ weights_chunk, None

    denominator_init = jnp.zeros(zs_batch.shape[0], dtype=jnp.float32)
    denominator, _ = jax.lax.scan(denom_body_fn, denominator_init, jnp.arange(num_full_chunks))

    # Remainder calc: computing denominator for image batch (zs_batch, cov_zs_batch), over last batch of nodes (zs_grid) if any leftover
    if remainder > 0:
        grid_chunk = zs_grid[num_full_chunks * batch_size :]
        weights_chunk = weights[num_full_chunks * batch_size :]
        log_likelihood_batch = -1 * ld.compute_latent_log_likelihood_no_batch(
            grid_chunk, zs_batch, cov_zs_batch, det_cov_zs_batch
        ).astype(jnp.float32)
        exp_norm_remainder = custom_exp_normalize(log_likelihood_batch, axis=1)
        denominator += exp_norm_remainder @ weights_chunk

    ## Second pass through zs_grid batches: computing the gradient summation, for the image batch
    # Computing gradient for image batch (zs_batch, cov_zs_batch), over all full batches of nodes (zs_grid)
    def grad_body_fn(grad_batch_accum, idx):
        grid_chunk = jax.lax.dynamic_slice_in_dim(zs_grid, idx * batch_size, batch_size, axis=0)
        log_likelihood_batch = -1 * ld.compute_latent_log_likelihood_no_batch(
            grid_chunk, zs_batch, cov_zs_batch, det_cov_zs_batch
        ).astype(jnp.float32)
        exp_norm = custom_exp_normalize(log_likelihood_batch, axis=1)
        val = jnp.sum(exp_norm / denominator[:, None], axis=0)
        grad_batch_accum = jax.lax.dynamic_update_slice(grad_batch_accum, val, (idx * batch_size,))
        return grad_batch_accum, None

    grad_batch_init = jnp.zeros(zs_grid.shape[0], dtype=jnp.float32)
    grad_batch, _ = jax.lax.scan(grad_body_fn, grad_batch_init, (jnp.arange(num_full_chunks)))

    # Remainder calc: computing grad for image batch (zs_batch), over last batch of confs (zs_grid) if any leftover
    if remainder > 0:
        val = jnp.sum(exp_norm_remainder / denominator[:, None], axis=0)
        grad_batch = grad_batch.at[num_full_chunks * batch_size :].set(val)

    return grad_batch, denominator.sum()


def compute_online_grad_and_loss(weights, zs, cov_zs, det_cov_zs, zs_grid, batch_size_zs=1000, batch_size_nodes=1000):
    """Gradient and loss computation, batched over zs (embedded images), and batched over zs_grid (latent volumes/confs).
    For the batching, a log likelihood matrix cannot be pre-computed, its pre-computed at iteration.
    This is a naive first try at this, maybe there is a way of caching some of these to not re-use on subsequent evaluations....
    This is a batched version of compute_grad()
    """
    ## Batching through zs, getting full batches and remainder batch
    num_data = zs.shape[0]
    num_full_chunks = num_data // batch_size_zs
    remainder = num_data % batch_size_zs

    ## Computing gradient over full batches
    grad = jnp.zeros(zs_grid.shape[0], dtype=jnp.float32)
    loss = 0
    for i in range(num_full_chunks):
        zs_chunk = zs[i * batch_size_zs : (i + 1) * batch_size_zs]
        cov_zs_chunk = cov_zs[i * batch_size_zs : (i + 1) * batch_size_zs]
        det_cov_zs_chunk = det_cov_zs[i * batch_size_zs : (i + 1) * batch_size_zs]
        grad_val, loss_val = compute_grad_and_loss_image_batch(
            weights, zs_chunk, cov_zs_chunk, det_cov_zs_chunk, zs_grid, batch_size=batch_size_nodes
        )
        grad = grad_accum(grad, grad_val, 1 / zs.shape[0])
        loss += -(1 / num_data) * jnp.log(loss_val)

    ## Computing gradient over last batch of images if any leftover
    if remainder > 0:
        zs_chunk = zs[num_full_chunks * batch_size_zs :]
        cov_zs_chunk = cov_zs[num_full_chunks * batch_size_zs :]
        det_cov_zs_chunk = det_cov_zs[num_full_chunks * batch_size_zs :]
        val = compute_grad_and_loss_image_batch(
            weights, zs_chunk, cov_zs_chunk, det_cov_zs_chunk, zs_grid, batch_size=batch_size_nodes
        )
        grad = grad_accum(grad, val, 1 / zs.shape[0])
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
    batch_size_zs=1000,
    batch_size_nodes=1000,
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
    # cov_zs = jnp.tile(jnp.eye(2), (cov_zs.shape[0], 1, 1)).astype(jnp.float32)*jnp.mean(jnp.exp(det_cov_zs[:, None, None]))**(1/pca_dim)
    # det_cov_zs = ld.compute_log_det_cov(cov_zs).astype(jnp.float32)

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

    ## Initialize scaling for gap stopping criteria, and pre-compute log likelihood matrix if not online
    if online:
        ## Compute initial gap scale with online gradients
        logger.info("Using online mode for saving memory, recomputing likelihoods per gradient iteration")
        grad_init, _ = compute_online_grad_and_loss(
            weights, zs, cov_zs, det_cov_zs, nodes, batch_size_zs=batch_size_zs, batch_size_nodes=batch_size_nodes
        )
        gap_scale = scaled_gap(grad_init, weights, scale=1.0)
    else:
        ## Compute full likelihood matrix, re-use in non-online (full) gradient updates
        log_likelihood = -1 * ld.compute_latent_log_likelihood(nodes, zs, cov_zs, det_cov_zs).astype(jnp.float32)
        likelihood = normalize_log_likeli_to_likeli(log_likelihood)
        (grad_init,) = compute_grad_and_loss(weights, likelihood)
        gap_scale = scaled_gap(grad_init, weights, scale=1.0)

    reached_gap = False

    losses = []
    gaps = []
    for k in range(max_iterations):
        ## Update grad and loss
        if online:
            grad, loss = compute_online_grad_and_loss(
                weights, zs, cov_zs, det_cov_zs, nodes, batch_size_zs=batch_size_zs, batch_size_nodes=batch_size_nodes
            )
        else:
            grad, loss = compute_grad_and_loss(weights, likelihood)
        losses.append(loss)

        ## Check stopping criterions
        gap = scaled_gap(grad, weights, gap_scale)
        gaps.append(gap)
        if k % 10 == 0:
            logger.info(f" iteration {k}, gap: {gap}")

        ## Check current gap against tolerance
        if not reached_gap and gap < tol:
            logger.info(f"reached gap tolerance, at iteration idx: {k}")
            logger.info(f"gap: {gap}")
            reached_gap = True

        # Check if stopping criteria met
        if reached_gap:
            logger.info(f"exiting! At iteration: {k}")
            break

        # Update weights
        weights = update_weights(weights, grad)

    # Return a grid of weights, and other info
    weights = weights.reshape((num_points_per_dim,) * pca_dim)
    losses = jnp.stack(losses)
    gaps = jnp.stack(gaps)
    return weights, losses, gaps


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


def plot_density(density, function=None, cmap="inferno", plots_dir=None):
    from recovar.output.output import sum_over_other

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

    density = np.asarray(density)

    if density.ndim == 1:
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs = np.atleast_1d(axs)
        axs[0].plot(density)
        axs[0].set_title("PC 0")
        axs[0].set_ylabel("MG density")
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
    else:
        n_pairs = density.ndim - 1
        n_extra = 1 if density.ndim > 2 else 0
        n_cols = n_pairs + n_extra

        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
        axs = np.atleast_1d(axs)

        col = 0
        for k in range(1, density.ndim):
            to_plot = function(density, [0, k])
            axs[col].imshow(to_plot.T, cmap=cmap, origin="lower")
            axs[col].set_title(f"PC x=0, y={k}")
            axs[col].set_xticklabels([])
            axs[col].set_yticklabels([])
            if col == 0:
                axs[col].set_ylabel("MG density")
            col += 1

        if density.ndim > 2:
            to_plot = function(density, [1, 2])
            axs[col].imshow(to_plot.T, cmap=cmap, origin="lower")
            axs[col].set_title("PC x=1, y=2")
            axs[col].set_xticklabels([])
            axs[col].set_yticklabels([])

    plt.subplots_adjust(wspace=0.05)
    if plots_dir is not None:
        plt.savefig(str(plots_dir / "density.png"))
    return fig, axs
