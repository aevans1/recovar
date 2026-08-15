"""Generate a synthetic dataset from pre-computed principal component mrcs"""
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
    pipeline_dir,
    image_size=128,
    dataset_params_option="uniform",
    latent_distribution_path=None,
    noise_level=0.1,
    noise_scale_std=0.0,
    contrast_std=0.0,
    n_images=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    grid_size = image_size

    #--------- Get voxel_size from input volumes
    # TODO: replace "volume_folder_input" below(folder with .mrcs), with a pipeline path, and get mean.mrc, and any number of pcs from an input pipeline, and let user pass it
    # TODO: just read out asset size from the volume size in the input volumes below
    logger.info("For now, hard coding what volumes are used, with no user choice, need to implement a possibility of loading from a pipeline instead")  
    logger.info("For now, 128^3 size volumes!! Check if loading volumes that are 256, needs to be changed if so")  
    asset_size = 128  # Needs to be size of the asset volumes used! TODO: just read this out from the volume size in the input volumes below
    volume_folder_input = f"/mnt/home/levans/software/recovar/recovar/assets/PCA_example_10345_downsampled_{asset_size}"
    output_folder = os.path.join(output_dir, "test_dataset")
    output.mkdir_safe(output_folder)
    n_images = int(n_images)

    # Voxel size scales with grid size to keep the same physical extent as the asset_size-px assets.
    voxel_size = 4.25 * asset_size / grid_size

    #---------Define PC space bounds
    # using a prob distribution on R^d 
    # TODO: don't hardcode 2 volumes
    # TODO: don't hardcode latent space bounds
    # TODO: don't hardcode num_poitns_per_dim, use code from elsewhere
    logger.info("For now, hardcoding dim=2 volumes")  
    logger.info("For now, hardcoding `latent space bounds'")  
    logger.info("For now, hardcoding num_points_per_dim=200, since hardcoding dim=2")  
    pca_dim = 2
    #latent_space_bounds = ld.compute_latent_space_bounds(zs, percentile=1)
    latent_space_bounds = np.array([[-1e3, 1e3], [-1e3, 1e3]])

    num_points_per_dim = 200
    num_nodes = num_points_per_dim**pca_dim

    #---------Load Volume distribution
    if latent_distribution_path is None:
        latent_distribution = np.ones(num_nodes)/num_nodes
        logger.info("using uniform distribution on latent space")
    else:
        latent_distribution = np.load(latent_distribution_path)
        logger.info("using loaded latent distribution on latent space")

    #---------Simulate
    simulator.generate_synthetic_dataset_mix_volumes(
         output_folder=output_folder,
         pipeline_dir=pipeline_dir,
         latent_space_bounds=latent_space_bounds,
         voxel_size=voxel_size,
         volumes_path_root=volume_folder_input,
         n_images=n_images,
         grid_size=grid_size,
         latent_distribution=latent_distribution,
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
    parser.add_argument("--pipeline-dir", type=str, default="", help="pipeline directory with a better explanation later")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (default: 64 for 64x64 images)")
    parser.add_argument("--latent-distribution-path", default=None, help="path to a volume distribution probability vector")
    parser.add_argument(
        "--volume-input", default=None, help="Optional input volume prefix (e.g. /path/to/vol for vol0000.mrc, ...)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible dataset generation")
    parser.add_argument("--dataset-params-option", type=str, default="dataset2", help="")
    parser.add_argument("--noise-scale-std", type=float, default=0.0, help="")
    parser.add_argument("--contrast-std", type=float, default=0.0, help="")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    make_test_dataset_from_PCs(
        output_dir=args.output_dir,
        pipeline_dir=args.pipeline_dir,
        image_size=args.image_size,
        dataset_params_option=args.dataset_params_option,
        latent_distribution_path=args.latent_distribution_path,
        noise_level=args.noise_level,
        noise_scale_std=args.noise_scale_std, 
        contrast_std=args.contrast_std,
        n_images=args.n_images,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
