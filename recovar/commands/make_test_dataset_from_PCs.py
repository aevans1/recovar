import argparse
import logging
import os

import numpy as np
import recovar.jax_config

from recovar.output import output
from recovar.simulation import simulator

logger = logging.getLogger(__name__)


def make_test_dataset_from_PCs(
    output_dir,
    image_size=64,
    dataset_params_option="dataset2",
    noise_level=0.1,
    noise_scale_std=0.0,
    contrast_std=0.1,
    n_images=None,
    seed=None,
    volume_input=None,
):
    """Generate a synthetic test dataset used by integration tests and examples.

    Parameters keep backward compatibility with older callers while also
    supporting newer CLI aliases:
    - ``grid_size``: alias of ``image_size`` (takes precedence when provided)
    - ``volume_input``: volume prefix root (default: bundled assets)
    - ``n_tilts``: number of tilts for ``tilt_series=True`` (default: 27)
    """
    if seed is not None:
        np.random.seed(seed)
    grid_size = image_size

    this_dir = os.path.dirname(__file__)
    volume_folder_input = volume_input if volume_input is not None else os.path.join(this_dir, "..", "assets", "vol")

    output_folder = os.path.join(output_dir, "test_dataset")
    output.mkdir_safe(output_folder)
    n_images = 1000 if n_images is None else int(n_images)
    # Voxel size scales with grid size to keep the same physical extent as the 128-px assets.
    voxel_size = 4.25 * 128 / grid_size

    # Historical default for bundled 3-volume assets; for custom volume sets,
    # use uniform distribution over however many volumes are provided.
    volume_distribution = np.array([1 / 4, 1 / 4, 1 / 2]) if volume_input is None else None

    image_stack, sim_info = simulator.generate_synthetic_dataset(
         output_folder=output_folder,
         voxel_size=voxel_size,
         volumes_path_root=volume_folder_input,
         n_images=n_images,
         grid_size=grid_size,
         volume_distribution=volume_distribution,
         dataset_params_option=dataset_params_option,
         noise_level=noise_level,
         noise_model="radial1",
         put_extra_particles=False,
         percent_outliers=0.0,
         volume_radius=0.7,
         trailing_zero_format_in_vol_name=True,
         noise_scale_std=noise_scale_std,
         contrast_std=contrast_std,
         disc_type="linear_interp",
     )
    logger.info("Finished generating dataset %s", output_folder)
    
def build_parser():
    """Making a separate parser file so that it's easier to generate configs later"""
    
    parser = argparse.ArgumentParser(description="Generate a test dataset for recovar")
    parser.add_argument("output_dir", nargs="?", default=os.getcwd(), help="Output directory for the test dataset")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (default: 64 for 64x64 images)")
    parser.add_argument(
        "--volume-input", default=None, help="Optional input volume prefix (e.g. /path/to/vol for vol0000.mrc, ...)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible dataset generation")
    parser.add_argument("--dataset-params-option", type=str, default="dataset2", help="")
    parser.add_argument("--noise-scale-std", type=float, default=0.0, help="")
    parser.add_argument("--contrast-std", type=float, default=0.1, help="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    make_test_dataset_from_PCs(
        output_dir=args.output_dir,
        image_size=args.image_size,
        dataset_params_option=args.dataset_params_option,
        noise_level=args.noise_level,
        noise_scale_std=args.noise_scale_std, 
        contrast_std=args.contrast_std,
        n_images=args.n_images,
        seed=args.seed,
        volume_input=args.volume_input,
    )

if __name__ == "__main__":
    main()
