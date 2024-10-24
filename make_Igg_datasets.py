import recovar.config
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
ftu = fourier_transform_utils(jnp)
from recovar import image_assignment, noise
from sklearn.metrics import confusion_matrix
from recovar import simulate_scattering_potential as ssp
from recovar import simulator, utils, image_assignment, noise, output, dataset
import prody
reload(simulator)
from scipy.stats import vonmises


import argparse
import sys
import os


def make_file(file_num, extension, leading_zeros):
    """
    Creates a string of type "000.ext" for extension and depending on number of leading zeros
    """
    file_string = format(file_num, f'0{leading_zeros}d') + extension
    return file_string

def load_pdbs_from_dir(pdb_folder, extension='.pdb', leading_zeros=3):
    """
    Reads a list of pdbs files of format like "000.pdb","001.pdb",etc, parses with prody, and throws in a list
    """
    idx =0 
    files = []
    pdb_string = pdb_folder + "/" + make_file(idx, extension=extension, leading_zeros=leading_zeros)
    print(pdb_string)
    while(os.path.isfile(pdb_string)):
        files.append(pdb_string)
        idx+=1
        pdb_string = pdb_folder + "/" + make_file(idx, extension=extension, leading_zeros=leading_zeros)
    pdb_atoms = [ prody.parsePDB(file) for file in files]
    return pdb_atoms

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", default=1000, type=int)
    parser.add_argument("--noise_lower_bound", default=-1, type=int)
    parser.add_argument("--noise_upper_bound", default=4, type=int)
    parser.add_argument("--noise_num_steps", default=10, type=int)
    parser.add_argument("--output_folder", default="/mnt/home/levans/ceph/igg_lukes", type=str)
    parser.add_argument("--pdb_folder", default ="/mnt/home/levans/ceph/igg/IgG-1D/pdbs", type=str)

    args = parser.parse_args()
    n_images = args.n_images
    output_folder = args.output_folder
    pdb_folder = args.pdb_folder
    noise_lower_bound = args.noise_lower_bound
    noise_upper_bound = args.noise_upper_bound
    noise_num_steps = args.noise_num_steps

    # Some imaging parameters to set
    grid_size = 256
    Bfactor = 60 
    noise_levels = np.logspace(noise_lower_bound, noise_upper_bound, noise_num_steps)
    voxel_size = 1.3 * 256 / grid_size
    volume_shape = tuple(3*[grid_size])
    disc_type_sim = 'nufft'

    # Load pdbs
    pdb_atoms = load_pdbs_from_dir(pdb_folder)

    # Shift atoms
    atoms = pdb_atoms[0]
    coords = atoms.getCoords()
    offset = ssp.get_center_coord_offset(coords)
    for atoms in pdb_atoms:
        atoms.setCoords(atoms.getCoords() - offset)
        
    # Make B-factored volumes (will be considered ground truth)     
    Bfaced_vols = len(pdb_atoms)*[None]
    for idx, atoms in enumerate(pdb_atoms):
        volume = ssp.generate_molecule_spectrum_from_pdb_id(atoms, voxel_size = voxel_size,  grid_size = grid_size, do_center_atoms = False, from_atom_group = True)
        Bfaced_vols[idx] = simulator.Bfactorize_vol(volume.reshape(-1), voxel_size, Bfactor, volume_shape)

    volume_folder = output_folder + '/' + 'simulated_test_volumes/'
    output.mkdir_safe(volume_folder)
    output.save_volumes( Bfaced_vols, volume_folder, from_ft= True)

    ## Define density that volumes are resampled from
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
    volume_distribution = p(x)
    volume_distribution /= (np.sum(volume_distribution))

    # First, simulate clean images, check norm of images
    noise_level = 
    dataset_folder = output_folder + '/' + f'dataset_clean/'
    image_stack, sim_info = simulator.generate_synthetic_dataset(dataset_folder, voxel_size, volume_folder, n_images,
        outlier_file_input = None, grid_size = grid_size,
        volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = noise_level,
        noise_model = "white", put_extra_particles = False, percent_outliers = 0.00, 
        volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0, contrast_std = 0, disc_type = disc_type_sim)
    print(image_stack.shape) 

    # Simulate images
    for idx, noise_level in enumerate(noise_levels):

        print(f"Starting at noise level {idx} of {len(noise_levels)}") 
        
        # Generate dataset
        noise_level = noise_level
        dataset_folder = output_folder + '/' + f'dataset{idx}/'
        image_stack, sim_info = simulator.generate_synthetic_dataset(dataset_folder, voxel_size, volume_folder, n_images,
            outlier_file_input = None, grid_size = grid_size,
            volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = noise_level,
            noise_model = "white", put_extra_particles = False, percent_outliers = 0.00, 
            volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0, contrast_std = 0, disc_type = disc_type_sim)
        
        dataset_options = dataset.get_default_dataset_option()
        dataset_options['particles_file'] = dataset_folder + '/' + f'particles.{grid_size}.mrcs'
        dataset_options['ctf_file'] = dataset_folder + '/' + f'ctf.pkl'
        dataset_options['poses_file'] = dataset_folder + '/' + f'poses.pkl'
        
        # Dump results to file
        recovar.utils.pickle_dump( sim_info, dataset_folder + '/' + 'sim_info.pkl')

    # Save info relevant to all datasets
    recovar.utils.pickle_dump( noise_levels, output_folder + '/' + 'noise_levels.pkl')

if __name__ == '__main__':
    main()
    print("Done")
