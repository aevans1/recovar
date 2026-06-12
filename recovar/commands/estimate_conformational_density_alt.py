import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.heterogeneity import calibrate_density
from recovar.output import output

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate conformational density from recovar results, trying a different technique")
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
        default=4,
        help="Dimension of PCA space in which the density is estimated (default 4). The runtime increases exponentially with this number, so <=5 is recommended.",
    )
    parser.add_argument(
        "--z_dim_used",
        type=int,
        default=None,
        help="Dimension of latent variable used (default smallest zdim stored >= pca_dim). Should be at least as big as pca_dim, and should be one of the dims used in analyze.py",
    )
    parser.add_argument(
        "--percentile_reject",
        type=int,
        default=10,
        help="Percentile of data to reject b/c they have large covariance (default 10%%)",
    )
    parser.add_argument(
        "--num_disc_points",
        type=int,
        default=None,
        help="Number of discretization points in each dimension for the grid density estimation. Default = 50 for dim >3, 100 for dim = 3, 200 for dim = 2",
    )
    parser.add_argument(
        "--percentile_bound",
        type=int,
        default=1,
        help="Rejects zs with coordinates above this bound for deciding the bounds of the grid (default 1 =1%%)",
    )
    return parser.parse_args()


def estimate_conformational_density_alt(
    recovar_result_dir,
    output_dir=None,
    pca_dim=4,
    z_dim_used=4,
    percentile_reject=10,
    num_points_per_dim=None,
    batch_size_zs=10000,
    batch_size_nodes=10000
):

    recovar_result_dir = Path(recovar_result_dir).expanduser().resolve()
    if not recovar_result_dir.exists():
        raise FileNotFoundError(f"recovar_result_dir {recovar_result_dir} does not exist")

    pipeline_output = output.PipelineOutput(str(recovar_result_dir))

    if z_dim_used is None:
        z_dim_all = np.asarray(pipeline_output.get("input_args").zdim)
        z_dim_all = z_dim_all[z_dim_all >= pca_dim]
        z_dim_used = np.min(z_dim_all)

    if pca_dim > z_dim_used:
        raise ValueError(f"pca_dim {pca_dim} should be less than or equal to z_dim_used {z_dim_used}")
    if pca_dim > 3:
        raise ValueError(f"pca_dim {pca_dim} should be less than or equal to 4. It will take very long for 4 dimension, at it's current implementation.")

    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else recovar_result_dir / "density_alt"
    output.mkdir_safe(str(output_dir))
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    output.mkdir_safe(str(plots_dir))
    output.mkdir_safe(str(data_dir))

    density = calibrate_density.multiplicative_gradient(
            pipeline_output,
            zdim=z_dim_used,
            noreg=True,
            pca_dim_max=pca_dim,
            percentile_reject=percentile_reject,
            num_points_per_dim=num_points_per_dim,
            tol=1e-6,
            max_iterations=10,
            batch_size_zs=batch_size_zs,
            batch_size_nodes=batch_size_nodes
        )
    )
    logger.info("Deconvolution done, size = %s", density.shape)
    calibrate_density.plot_density(density)
    plt.savefig(str(plots_dir / "density.png"))
    plt.close()

    # TODO: for plotting stopping criteria curves, etc
    #plt.figure(figsize=(12, 10))
    #for i, (alpha, c) in enumerate(zip(alphas, cost)):
    #    plt.text(alpha, c, str(i), fontsize=18)
    #plt.loglog(alphas, cost, "-o")
    #plt.loglog(np.ones(2) * alphas[knee_idx], [min(cost), max(cost)], "--", color="black")
    #plt.text(
    #    alphas[knee_idx],
    #    min(cost),
    #    f"knee point: {alphas[knee_idx]:.2e}, idx={knee_idx}",
    #    rotation=90,
    #    verticalalignment="bottom",
    #)
    #plt.ylabel("Cost")
    #plt.xlabel("Lambda (regularization parameter)")
    #plt.gca().invert_xaxis()
    #plt.savefig(str(plots_dir / "Lcurve.png"), transparent=True)
    #plt.close()


def main():
    args = parse_args()

    from recovar.project.job_context import job_context

    with job_context(args, "estimate_conformational_density") as ctx:
        result_dir = ctx.pipeline_dir or args.recovar_result_dir
        estimate_conformational_density(
            recovar_result_dir=result_dir,
            output_dir=ctx.output_dir,
            pca_dim=args.pca_dim,
            z_dim_used=args.z_dim_used,
            percentile_reject=args.percentile_reject,
            num_disc_points=args.num_disc_points,
            alphas=args.alphas,
            percentile_bound=args.percentile_bound,
        )


if __name__ == "__main__":
    main()
