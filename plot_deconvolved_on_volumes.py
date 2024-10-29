from recovar import output, synthetic_dataset, metrics, simulator, utils
import json
import os
import numpy as np
import recovar.latent_density as ld
import matplotlib.pyplot as plt
from scipy.stats import vonmises
import pickle

## NOTE: the dataset folder has to matched up correctly with the recovar result directory!
recovar_result_dir = '/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_simulated_mid_snr'
dataset_folder = '/mnt/home/levans/ceph/igg_lukes/dataset4'

## Pick dimension of whatever deconvolved embedding we are looking at
zdim = 2
density_dir = recovar_result_dir + "/" + f'density_{zdim}'

## Gets embedding from ground truth volumes
# For now: making a custom projection that doesn't load in the whole PC matrix U, full matrix is too large
#zs_gt = metrics.get_gt_embedding_from_projection(volumes, pipeline_output.get('u'), pipeline_output.get('mean'))
zs_gt_fname = density_dir + "/" + f"embedded_gt_volumes_zdim{zdim}.npy"
if os.path.isfile(zs_gt_fname):
    zs_gt = np.load(zs_gt_fname)
else:
    volumes_path_root = '/mnt/home/levans/ceph/igg_lukes/simulated_test_volumes'
    pipeline_output = output.PipelineOutput(recovar_result_dir)
   
    ## load volumes
    def make_file(k):
        return volumes_path_root + "/" + format(k, '04d')+".mrc"
    idx =0 
    files = []
    while(os.path.isfile(make_file(idx))):
        files.append(make_file(idx))
        idx+=1
    volumes, _ = simulator.generate_volumes_from_mrcs(files, None, padding= 0 )
    
    ## For simulated data: volumes need to be rescaled according to recovars simulator
    file = open(dataset_folder + '/' + 'sim_info.pkl', 'rb')
    scale_vol = pickle.load(file)['scale_vol']
    volumes *= scale_vol
    
    ## Old volume scaling here
    #print(f"mean of mean vol pixels: {np.mean(mean)}")
    #print(f"mean of gt vol pixels: {np.mean(volumes)}")
    #print(np.mean(mean).real / np.mean(volumes))
    #scale = np.mean(mean) / np.mean(volumes)
    #volumes *= scale**(0.5)
    
    zs_gt = (np.conj(pipeline_output.get('u'))[:zdim, :] @ (volumes - pipeline_output.get('mean')).T).T.real
    np.save(density_dir + "/" + f"embedded_gt_volumes_zdim{zdim}.npy", zs_gt)

## grab some other embedded data to compare with
zs = np.loadtxt(recovar_result_dir + '/' + 'analysis_4/kmeans_center_coords.txt')


figure_dir = density_dir + '/' + 'figures'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

## Visually check that pc embeddings of volumes line up with some embedded kmeans clusters
for i in range(zdim):
    for j in range(i+1,zdim):
        plt.figure()
        plt.scatter(zs[:, i], zs[:, j], s= 2, label="kmeans cluster centers")
        plt.scatter(zs_gt[:, i], zs_gt[:, j], s = 2, label="embedded volumes")
        plt.xlabel(f"PC{i}")
        plt.ylabel(f"PC{j}")
        plt.legend()
        plt.savefig(figure_dir + '/' + f"latent_vols_kmeans_plot_PC{i}{j}.png", dpi=300)

### define density that volumes were resampled from
def p(x):
    means = [np.pi/2, np.pi, 3*np.pi/2]
    kappas =  [6.0, 6.0, 6.0]
    weights = np.array([2.0, 1.0, 2.0])
    weights /= sum(weights)  
    val = 0
    for i in range(3): 
        val += weights[i]*vonmises.pdf(x, loc=means[i], kappa=kappas[i])
    return val
x = np.linspace(0, 2*np.pi, 100)
y = p(x)
y /= (np.sum(y))

# Plot ground truth density against raw density
density_file = utils.pickle_load(density_dir + "/" + f'all_densities/raw_density.pkl')
computed_deconvolve_density = density_file['density']
density_bounds = density_file['latent_space_bounds']
raw_density_at_zs_gt = output.density_on_grid(zs_gt, computed_deconvolve_density, density_bounds)
raw_density_at_zs_gt = np.array(raw_density_at_zs_gt)
raw_density_at_zs_gt /= np.sum(raw_density_at_zs_gt)
plt.figure()
plt.plot(x, y, linewidth=1.0, label="ground truth density", color='k', linestyle="dashed")
plt.plot(x, raw_density_at_zs_gt, label="raw density", linewidth=1.0)
plt.xlabel(r"Dihedral Angle($\degree$)", fontsize=16)
plt.ylabel("Probability",fontsize=16)
plt.legend()
plt.savefig(figure_dir + "/" + f"raw_density_at_gt_vols.png", dpi=300)

## Plot ground truth density against deconvolved densities
for k in range(11):

    # Load pre-computed density info
    density_file = utils.pickle_load(density_dir + '/' + f'all_densities/deconv_density_{k}.pkl')
    computed_deconvolve_density = density_file['density']
    density_bounds = density_file['latent_space_bounds']

    # Interpolate volumes to grid and return density there 
    density_at_zs_gt = output.density_on_grid(zs_gt, computed_deconvolve_density, density_bounds)
    density_at_zs_gt = np.array(density_at_zs_gt)
    density_at_zs_gt /= np.sum(density_at_zs_gt)

    plt.figure() 
    plt.plot(x, y, linewidth=1.0, label="ground truth density", color='k', linestyle='dashed')
    plt.plot(x, raw_density_at_zs_gt, label="raw density", linewidth=1.0)
    plt.plot(x, density_at_zs_gt, label="deconvolved density", linewidth=1.0)
    plt.xlabel(r"Dihedral Angle($\degree$)", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend()
    plt.savefig(figure_dir + "/" + f"deconv_density_at_gt_vols_{k}.png", dpi=300)


# replot for d


##### old code for loading in paths instead of ground truth volumes above
#output_dir = '/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_high_snr/analysis_10/path0/'
#path_json = json.load(open(output_dir + '/path.json', 'r'))
#density = path_json['density']
#path = path_json['path']
#zs_gt = np.array(path)

#old_grid = np.linspace(0, 2*np.pi, len(zs_gt))
#zs_gt_plot = np.interp(x, old_grid, zs_gt)
#output_dir = '/mnt/home/levans/Projects/Model_bias_heterogeneity/Igg/output_high_snr/analysis_10/path3/'
#path_json = json.load(open(output_dir + '/path.json', 'r'))
#density = path_json['density']
#path = path_json['path']
#zs_gt = np.concatenate([zs_gt, np.array(path)])
#
#print("trying to replace gt vols with a path")
#
#plt.figure()
#plt.scatter(zs[:, 0], zs[:, 1], s= 2, label="kmeans cluster centers")
#plt.scatter(zs_gt[:, 0], zs_gt[:, 1], s = 2, label="embedded path")
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.legend()
#plt.savefig("scaling_issue_plot_path.png")
#

