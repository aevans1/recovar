"""Generate a synthetic dataset from pre-computed principal component mrcs"""
import argparse
import logging
import os

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
 
from recovar.output import output
from recovar.simulation import simulator
from recovar.heterogeneity import latent_density as ld

logger = logging.getLogger(__name__)


def make_test_dataset_from_PCs(
    output_dir,
    pipeline_dir=None, 
    asset_size=128,
    image_size=128,
    dataset_params_option="uniform",
    latent_generation_option="samples",
    latent_distribution_path=None,
    latent_distribution_params_option="two_channels",
    noise_level=0.1,
    noise_scale_std=0.0,
    contrast_std=0.0,
    n_images=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    grid_size = image_size

    if pipeline_dir is None:
        ## Load everything from pre-computed assets, PCs from empiar 10345, and load pipeline for getting latent zs
        if asset_size == 128:
            volume_folder_input = f"/mnt/home/levans/software/recovar/recovar/assets/PCA_example_10345_downsampled_128"
            pipeline_dir = "/mnt/home/levans/ceph/recovar_testing/recovar_on_recovar_datasets/10345_downsampled_128"
        elif asset_size == 256:
            volume_folder_input = f"/mnt/home/levans/software/recovar/recovar/assets/PCA_example_10345"
            pipeline_dir = "/mnt/home/levans/ceph/recovar_testing/recovar_on_recovar_datasets/10345"

    else:
        # TODO: replace "volume_folder_input" below(folder with .mrcs), with a pipeline path, and get mean.mrc, and any number of pcs from an input pipeline, and let user pass it
        #       Then, just read out asset size from the volume size in the input volumes below
        raise NotImplementedError("Custom pipelines, with pcs to simulate from...not implemented yet")

    output_folder = os.path.join(output_dir, "test_dataset")
    output.mkdir_safe(output_folder)
    n_images = int(n_images)

    # Voxel size scales with grid size to keep the same physical extent as the asset_size-px assets.
    voxel_size = 4.25 * asset_size / grid_size

    #---------Define PC space bounds
    # TODO: don't hardcode 2 volumes, let it be variable
    # TODO: don't hardcode num_points_per_dim, use code from elsewhere that determines this
    logger.info("For now, hardcoding dim=2 volumes")  
    logger.info("For now, hardcoding `latent space bounds'")  
    logger.info("For now, hardcoding num_points_per_dim=200, since hardcoding dim=2, and 200 is standard elsewhere in library")  
    pca_dim = 2

    logger.info("For now, loaded pipeline_dir is hardcoded as same dir as volume folder input, in future need to load this both from pipeline")
    po = output.PipelineOutput(pipeline_dir)
    zs_noreg = po.get_embedding_component('latent_coords_noreg', pca_dim)
    latent_space_bounds = ld.compute_latent_space_bounds(zs_noreg, percentile=1)
    logger.info(f"original pipeline latent_space_bounds:{latent_space_bounds}")

    num_points_per_dim = 200
    num_nodes = num_points_per_dim**pca_dim

    if latent_generation_option == "samples":
        # "samples": gets the latents from sampling, for true "continuous" heterogeneity in the latent space

        ##---------Simulate latent distribution and image assignments
        if latent_distribution_params_option == "gaussian":
            mean = jnp.zeros(2)
            scale = 1/4  # empirically this makes it so that the PC deviations are 4 stds away
            std = scale*(latent_space_bounds[:, 1] - latent_space_bounds[:, 0])/2
            key = jax.random.key(29838)
            image_assignments = jax.random.normal(key, shape=(n_images, std.shape[0])) * std + mean

        elif latent_distribution_params_option == "two_channels":
            # Simulate from langevin

            def logdensity_fn(v, beta=1.0):
                x = v[0]
                y = v[1]

                exp1 = jnp.exp(-12*(x**2 + y**2)) 
                exp2 = jnp.exp(-12*((x+0.5)**2 + y**2)) 
                exp3 = jnp.exp(-12*((x-0.5)**2 + y**2)) 
                poly1 = x**6
                poly2 = y**6
                U = 2*exp1 - exp2 - exp3 + poly1 + poly2
                return -beta*U

            image_assignments = simulate_image_assignments(n_images, logdensity_fn, beta=1.0)

        # scale image assignments to have roughly same bounds as the original zs 
        bounds_image_assignments = ld.compute_latent_space_bounds(image_assignments, percentile=1)
        rescale = (latent_space_bounds[:, 1] - latent_space_bounds[:, 0]) / (bounds_image_assignments[:, 1] - bounds_image_assignments[:, 0]) 
        image_assignments = rescale * image_assignments 
        bounds_image_assignments = ld.compute_latent_space_bounds(image_assignments, percentile=1)
        logger.info(bounds_image_assignments)
        logger.info(latent_space_bounds)

    elif latent_generation_option == "grid":
        # "grid": samples the latents on the same grid, roughly, that will be used for estimating the density. essentially a histogram
        
        if latent_distribution_path is None:
            latent_distribution = np.ones(num_nodes)/num_nodes
            logger.info("using uniform distribution on latent space")
        else:
            latent_distribution = np.load(latent_distribution_path)
            logger.info("using loaded latent distribution on latent space")


        pca_dim = 2
        num_points_per_dim = 200
        
        logger.info("For latent grid, transposing to match conventions used to make the input volume distribution") 
        grids_flat = ld.make_latent_space_grid_from_bounds(latent_space_bounds, num_points_per_dim).astype(jnp.float32)
        nodes = grids_flat.reshape(num_points_per_dim**pca_dim, grids_flat.shape[-1])

        seed = 8393939
        choice_key = jax.random.key(seed)

        image_assignments_indices = jax.random.choice(choice_key, jnp.arange(len(nodes)), shape=(n_images,), p=latent_distribution, mode="high").astype(int)
        image_assignments = nodes[image_assignments_indices]


    plt.scatter(image_assignments[:, 0], image_assignments[:, 1], s=1.0)
    plt.savefig(f"checking_initial_assignments_{latent_generation_option}.png", dpi=300)

    #---------Simulate
    simulator.generate_synthetic_dataset_mix_volumes(
         output_folder=output_folder,
         latent_space_bounds=latent_space_bounds,
         voxel_size=voxel_size,
         volumes_path_root=volume_folder_input,
         n_images=n_images,
         grid_size=grid_size,
         image_assignments=image_assignments,
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

def simulate_image_assignments(n_images, logdensity_fn, beta):
    rng_key = jax.random.key(32828282)
    dt = jnp.float32(1e-3)
    beta = jnp.float32(beta)

    run_langevin = _construct_langevin(logdensity_fn) 
    x_init = jnp.zeros(2, dtype=jnp.float32)
    image_assignments = run_langevin(rng_key, x_init, beta, dt=dt, num_steps=int(n_images*100))
    image_assignments = image_assignments[::100]
    return image_assignments

def _construct_langevin(logdensity_fn):

    def langevin_step(key, x, beta, dt):
            grad = jax.grad(lambda v: logdensity_fn(v, beta))(x)
            assert grad.dtype == jnp.float32, grad.dtype # We need everything here to be float32, so checking if the logdensity puts that in somewhere
            noise = jax.random.normal(key, shape=x.shape, dtype=jnp.float32)
            return x + (1/beta)*grad * dt + jnp.sqrt(2 * dt / beta) * noise

    @eqx.filter_jit
    def run_langevin(key, x_init, beta, dt, num_steps):
        def body(x, key):
            key, subkey = jax.random.split(key)
            x_update = langevin_step(subkey, x, beta, dt)
            return x_update, x_update
        keys = jax.random.split(key, num_steps) # make keys for all steps
        _, states = jax.lax.scan(lambda x, k: body(x, k), x_init, keys)
        return states

    return run_langevin


def build_parser():
    """Making a separate parser file so that it's easier to generate configs later"""
    
    parser = argparse.ArgumentParser(description="Generate a test dataset for recovar")
    parser.add_argument("output_dir", nargs="?", default=os.getcwd(), help="Output directory for the test dataset")
    parser.add_argument("--pipeline-dir", default=None, help="pipeline directory with a better explanation later")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--asset-size", type=int, help="Voxel size of assets used to generate")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (default: 64 for 64x64 images)")
    parser.add_argument("--latent-generation-option", type=str, default="samples", help="how to generate the latent vectors")
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
        asset_size=args.asset_size,
        image_size=args.image_size,
        dataset_params_option=args.dataset_params_option,
        latent_generation_option=args.latent_generation_option,
        latent_distribution_path=args.latent_distribution_path,
        noise_level=args.noise_level,
        noise_scale_std=args.noise_scale_std, 
        contrast_std=args.contrast_std,
        n_images=args.n_images,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
