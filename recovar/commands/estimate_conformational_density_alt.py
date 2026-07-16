import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recovar.heterogeneity import calibrate_density
from recovar.output import output
import recovar.utils as utils

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate conformational density from recovar results, trying a different technique"
    )
    parser.add_argument(
        "recovar_result_dir", type=str, help="Directory containing recovar results provided to pipeline.py"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the density estimation results. Default = recovar_result_dir/density/",
    )
    from recovar.utils.parser_args import add_output_name_arg, add_project_arg

    add_project_arg(parser)
    add_output_name_arg(parser)
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=2,
        help="Dimension of PCA space in which the density is estimated (default 2). <=4 is recommended.",
    )
    parser.add_argument(
        "--z_dim_used",
        type=int,
        default=None,
        help="Dimension of the embeddings to load in (default smallest z_dim_used stored >= pca_dim). Should be at least as big as pca_dim, and should be one of the dims used in analyze.py",
    )
    parser.add_argument(
        "--percentile_reject",
        type=int,
        default=10,
        help="Percentile of data to reject b/c they have large covariance (default 10%%)",
    )
    parser.add_argument(
        "--num_points_per_dim",
        type=int,
        default=None,
        help="Number of discretization points in each dimension for the grid density estimation. Default = 50 for dim >3, 100 for dim = 3, 200 for dim = 2",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Sets stopping tolerance for density estimation."   
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Sets max number of iterations for density estimation."   
    )
    parser.add_argument(
        "--online",
        action="store_true",
        default=False,
        help="If passed, recomputes likelihood at each gradient iteration, to save memory. Use if large number of images or nodes.",
    )
    parser.add_argument(
        "--weights_frequency",
        type=int,
        default=0,
        help="Sets frequency that intermediate weight outputs are saved in density estimation. At default of 0, no intermediates are saved."   
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=False,
        help="If true, density estimation runs to max iterations.",
    )
    parser.add_argument(
        "--batch_size_zs",
        type=int,
        default=None,
        help="Batch size of embedded images (zs) to use in online density estimation algorithm.",
    )
    parser.add_argument(
        "--batch_size_nodes",
        type=int,
        default=None,
        help="Batch size of nodes (grid points) to use in online density estimation algorithm.",
    )
    return parser.parse_args()


def estimate_conformational_density_alt(
    recovar_result_dir,
    output_dir=None,
    pca_dim=4,
    z_dim_used=4,
    percentile_reject=10,
    num_points_per_dim=None,
    tol=1e-4,
    max_iterations=1000,
    online=True,
    weights_frequency=0,
    diagnostic=False,
    batch_size_zs=10000,
):
    logger.info(f"online? {online}") 
    recovar_result_dir = Path(recovar_result_dir).expanduser().resolve()
    if not recovar_result_dir.exists():
        raise FileNotFoundError(f"recovar_result_dir {recovar_result_dir} does not exist")

    pipeline_output = output.PipelineOutput(str(recovar_result_dir))

    if z_dim_used is None:
        z_dim_used_all = np.asarray(pipeline_output.get("input_args").zdim)
        z_dim_used_all = z_dim_used_all[z_dim_used_all >= pca_dim]
        z_dim_used = np.min(z_dim_used_all)

    if pca_dim > z_dim_used:
        raise ValueError(f"pca_dim is {pca_dim}, should be less than or equal to z_dim_used {z_dim_used}")
    if pca_dim > 3:
        logger.info(
            f"pca_dim is {pca_dim}, should be less than or equal to 4. It is set larger than 3, and it will take very long for 4 dimensions, at it's current implementation."
        )

    output_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else recovar_result_dir / "density_alt"
    )
    output.mkdir_safe(str(output_dir))
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    output.mkdir_safe(str(plots_dir))
    output.mkdir_safe(str(data_dir))

    density, info = calibrate_density.multiplicative_gradient(
        pipeline_output,
        pca_dim=pca_dim,
        noreg=True,
        z_dim_used=z_dim_used,
        percentile_reject=percentile_reject,
        num_points_per_dim=num_points_per_dim,
        tol=tol,
        max_iterations=max_iterations,
        online=online,
        weights_frequency=weights_frequency,
        diagnostic=diagnostic,
        batch_size_zs=batch_size_zs,
    )
    losses = info["losses"]
    gaps = info["gaps"]
    weights_all = info["weights_all"]
    idx_weights = info["idx_weights"]
    idx_weights_gap = info["gap_idx"]
    weights_gap = info["weights_gap"]
    row_labels = [f"iteration {idx} " for idx in idx_weights]
    logger.info("Deconvolution done, size = %s", density.shape)

    all_densities_dir = data_dir / "all_densities"
    output.mkdir_safe(str(all_densities_dir))
    for idx, weights in enumerate(weights_all):
        utils.pickle_dump(
            {"density": weights, "latent_space_bounds": [], "iteration": idx_weights[idx]},
            str(all_densities_dir / f"deconv_density_alt_{idx}.pkl"),
        )

    utils.pickle_dump(
        {"density": weights_gap, "latent_space_bounds": [], "iteration": idx_weights_gap},
        str(data_dir / "deconv_density_alt_gap.pkl"),
    )


    calibrate_density.plot_density(weights_gap, plots_dir=plots_dir, cbar_normalize=False)
    calibrate_density.plot_density(weights_all, plots_dir=plots_dir, row_labels=row_labels, cbar_normalize=False)
    calibrate_density.plot_info(losses, gaps, plots_dir=plots_dir)
    plt.close()


def main():
    args = parse_args()

    from recovar.project.job_context import job_context

    with job_context(args, "estimate_conformational_density_alt") as ctx:
        result_dir = ctx.pipeline_dir or args.recovar_result_dir
        estimate_conformational_density_alt(
            recovar_result_dir=result_dir,
            output_dir=ctx.output_dir,
            pca_dim=args.pca_dim,
            z_dim_used=args.z_dim_used,
            percentile_reject=args.percentile_reject,
            num_points_per_dim=args.num_points_per_dim,
            tol=args.tol,
            max_iterations=args.max_iterations,
            online=args.online,
            weights_frequency=args.weights_frequency,
            diagnostic=args.diagnostic,
            batch_size_zs=args.batch_size_zs,
        )


if __name__ == "__main__":
    main()
