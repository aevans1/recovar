import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import deconvolve_density, latent_density, utils
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)
from recovar import parser_args
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import scipy
import scipy.interpolate as scinterp


def add_args(parser: argparse.ArgumentParser):

    parser = parser_args.standard_downstream_args(parser, analyze = True)

    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable (a single int, not a list)"
    )
    return parser

def deconvolve_latent_density(recovar_result_dir, output_folder = None, zdim = 4, no_z_reg = True):
    po = o.PipelineOutput(recovar_result_dir + '/')
 
    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    
    elif zdim is None:
        zdim = list(po.get('zs').keys())[0]
        logger.info(f"using zdim={zdim}")
    zdim_key = f"{zdim}_noreg" if no_z_reg else zdim

    if output_folder is None:
        output_folder = recovar_result_dir + f'/output/analysis_{zdim_key}/' 

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    # Make folder for outputs
    output_folder_deconv_density = output_folder + 'deconv_density/'
    o.mkdir_safe(output_folder_deconv_density)
    save_file = output_folder_deconv_density + 'results.pkl'


    alphas = np.flip(np.logspace(-6, 3, 10)) if alphas is None else alphas
    # Do the stuf
    lbfgsb_sols, alphas, cost, reg_cost, density, total_covar, grids, bounds = deconvolve_density.get_deconvolved_density(po, zdim='2_noreg', pca_dim_max=2, alphas=alphas, save_to_file=save_file)
    deconvolve_density.plot_density(lbfgsb_sols, density, alphas)

    plt.show()


def try_new_stuff(recovar_result_dir, output_folder = None, zdim = 4, no_z_reg = True, pca_dim_max = 2, percentile_reject=10):
    po = o.PipelineOutput(recovar_result_dir + '/')
 
    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

    zs = po.get('zs')[zdim]
    cov_zs =  po.get('cov_zs')[zdim]
    cov_zs_norm = np.linalg.norm(cov_zs, axis=(-1,-2), ord = 2)
    good_zs = cov_zs_norm >np.percentile(cov_zs_norm, percentile_reject)
    
    zs = zs[good_zs][:,:zdim]
    cov_zs = cov_zs[good_zs][:,:zdim,:zdim]

    import json
    with open('/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_vonmises/analysis_10/path0/path.json') as f:
        data = json.load(f)
        path = np.array(data["path"])
    print(path.shape)
    #centers = np.loadtxt("/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_adjusted_test/output/analysis_2/centers.txt")
    #print(centers.shape) 
    #centers = centers[:, :2]
    print(f"path shape: {path.shape}")
    print(f"zs shape: {zs.shape}")
    log_likelihood = latent_density.compute_latent_log_likelihood(path, zs, cov_zs)
    #log_likelihood = compute_latent_log_likelihood(path, zs, cov_zs)

    np.save("log_likelihood_centers.npy", log_likelihood)
    np.savetxt("log_likelihood_centers.txt", log_likelihood)
    np.save("path.npy", path)

def plot_deconv_on_path(recovar_result_dir, output_folder = None, zdim = 4, no_z_reg = True, pca_dim_max = 2, percentile_reject=10):
    po = o.PipelineOutput(recovar_result_dir + '/')
 
    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --zdim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    
    elif zdim is None:
        zdim = list(po.get('zs').keys())[0]
        logger.info(f"using zdim={zdim}")
    zdim_key = f"{zdim}_noreg" if no_z_reg else zdim

    if output_folder is None:
        output_folder = recovar_result_dir + f'/output/analysis_{zdim_key}/' 

    # Load up folder with deconvolution stuff 
    output_folder_deconv_density = output_folder + 'deconv_density/'
    save_file = output_folder_deconv_density + 'results.pkl'

    # Load up deconv info 
    deconvolve_info = utils.pickle_load(save_file)
    grids_inp = deconvolve_info['grids']
    deconv_densities = deconvolve_info['deconv_densities']
    alphas = deconvolve_info['alphas']

    # Load up a favorite path
    import json
    with open('/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_adjusted_test/output/analysis_2/path1/path.json') as f:
        data = json.load(f)
        path = np.array(data["path"])


    # Write out the density everything was generated from
    def p(x):
        means = [np.pi/2, np.pi, 3*(np.pi/2)]
        vars = [(np.pi/6)**2, ((np.pi/6)**2)/(1.5), (np.pi/6)**2]
        weights = np.array([1, 0.5, 1] )
        weights /= sum(weights)  
        val = 0
        for i in range(3): 
            val += weights[i]*(2*np.pi*vars[i])**(-0.5)*np.exp(-(x - means[i])**2 / (2*vars[i]))
        return val

    # Plot that density, normalize
    #x = np.linspace(0, 2*np.pi, path.shape[0])
    print("NOTE: plotting true density from pi/2 to 3pi/2 since the path it got is not quite matching up")
    x = np.linspace(np.pi/2, 3*np.pi/2, path.shape[0])
    y = p(x)
    y /= (np.sum(y))

    # Plot each recovered density, on the  "true density"
    for idx, alpha in enumerate(alphas): 
        density = deconv_densities[idx]
        grids = grids_inp.reshape(2500, 2)
        density_new = density.reshape(2500)
        interp = scinterp.NearestNDInterpolator(grids, density_new) 
        density_traj = interp(path) 
        #density_traj = evaluate_function_off_grid(density, path)
        density_traj /=np.sum(density_traj)
        plt.figure()
        plt.plot(x, y, color='k')
        plt.plot(x, density_traj)
        plt.savefig(f"density_traj_alpha_{idx}.jpg")
    plt.show()

def compute_latent_log_likelihood(test_pts, zs, cov_zs):
    assert zs.shape[1] == test_pts.shape[1]
    assert zs.shape[1] == cov_zs.shape[1]
    assert test_pts.ndim == 2
    assert cov_zs.ndim == zs.ndim + 1

    #quads = np.zeros([zs.shape[0], test_pts.shape[0]] )
    n_images = zs.shape[0]
    n_test_points = test_pts.shape[0]

    log_likelihood = jnp.zeros((n_images, n_test_points))
    for k in range(n_images):
        for j in range(n_test_points):
            covar_data = jnp.linalg.pinv(cov_zs[k])
            log_likelihood.at[k, j].set(jnp.log(jax.scipy.stats.multivariate_normal.pdf(test_pts[j, :], zs[j, :], covar_data)))
    return log_likelihood

def estimate_kernel_by_sampling_no_kde(grids_inp, cov_zs, num_samples = 5000):

    grid_size = jnp.max(grids_inp, axis = np.arange(grids_inp.ndim-1))  - jnp.min(grids_inp, axis = np.arange(grids_inp.ndim-1)) 
    coord_pca_1D = []
    num_points = grids_inp.shape[0]
    
    pca_dim_max = grids_inp.shape[-1]
    for pca_dim in range(pca_dim_max):
        coord_pca = jnp.flip(jnp.linspace(- grid_size[pca_dim]/2, grid_size[pca_dim]/2, num_points, endpoint = False))
        coord_pca_1D.append(coord_pca)
    grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
    grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 

    kernel_on_grid =0
    for k in range(num_samples):
        idx = np.random.choice(cov_zs.shape[0], 1)[0]
        covar_data = jnp.linalg.pinv(cov_zs[idx])
        total_covar = covar_data
        kernel_on_grid += jax.scipy.stats.multivariate_normal.pdf(grids_flat, np.zeros(total_covar.shape[0]), total_covar)

    kernel_on_grid = kernel_on_grid/jnp.sum(kernel_on_grid)

    return kernel_on_grid.reshape(grids_inp.shape[:-1])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    #deconvolve_latent_density(args.result_dir, output_folder=args.outdir)
    #plot_deconv_on_path(args.result_dir, output_folder=args.outdir)
    try_new_stuff(args.result_dir, output_folder=args.outdir, zdim=args.zdim)