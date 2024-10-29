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

import argparse
import sys



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", default=10000, type=int)
    parser.add_argument("--output_folder", default="/mnt/home/levans/ceph/spike/recovar_experiments", type=str)
    parser.add_argument("--pdb_folder", default="/mnt/home/levans/ceph/spike/spike_models_for_bad_histogram/all_atom/without_glycans", type=str)

    args = parser.parse_args()
    n_images = args.n_images
    output_folder = args.output_folder
    pdb_folder = args.pdb_folder

    # Some parameters to set
    grid_size = 256
    Bfactor = 60 
    noise_levels = np.logspace(-1,4,10)
    noise_levels = [0.1]
    ## Make volumes from PDB
    #pdbs = [ '3down_nogly.pdb', '1up_3drest_nogly.pdb']

    voxel_size = 1.3 * 256 / grid_size
    volume_shape = tuple(3*[grid_size])

    ## Center atoms (but shift by same amount)
    #pdb_atoms = [ prody.parsePDB(pdb_folder + '/' +  pdb_i) for pdb_i in pdbs ]
    #atoms =pdb_atoms[0]
    #coords = atoms.getCoords()
    #offset = ssp.get_center_coord_offset(coords)
    ## coords = coords - offset
    #for atoms in pdb_atoms:
    #    atoms.setCoords(atoms.getCoords() - offset)
        
    ## Make B-factored volumes (will be considered g.t.)     
    #Bfaced_vols = len(pdbs)*[None]
    #for idx, atoms in enumerate(pdb_atoms):
    #    volume = ssp.generate_molecule_spectrum_from_pdb_id(atoms, voxel_size = voxel_size,  grid_size = grid_size, do_center_atoms = False, from_atom_group = True)
    #    Bfaced_vols[idx] = simulator.Bfactorize_vol(volume.reshape(-1), voxel_size, Bfactor, volume_shape)

    disc_type_sim = 'nufft'

    volume_folder = output_folder + '/' + 'true_volumes/'
    #output.mkdir_safe(volume_folder)
    #output.save_volumes( Bfaced_vols, volume_folder, from_ft= True)

    for idx, noise_level in enumerate(noise_levels):

        print(f"Starting at noise level {idx} of {len(noise_levels)}") 
        # Generate dataset
        volume_distribution = np.array([0.8,0.2])
        noise_level = noise_level
        dataset_folder = output_folder + '/' + f'dataset{idx}/'
        image_stack, sim_info = simulator.generate_synthetic_dataset(dataset_folder, voxel_size, volume_folder, n_images,
            outlier_file_input = None, grid_size = grid_size,
            volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = noise_level,
            noise_model = "white", put_extra_particles = False, percent_outliers = 0.00, 
            volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0, contrast_std = 0, disc_type = disc_type_sim)
        
        #dataset_options = dataset.get_default_dataset_option()
        #dataset_options['particles_file'] = dataset_folder + '/' + f'particles.{grid_size}.mrcs'
        #dataset_options['ctf_file'] = dataset_folder + '/' + f'ctf.pkl'
        #dataset_options['poses_file'] = dataset_folder + '/' + f'poses.pkl'
        
        ## Dump results to file
        #recovar.utils.pickle_dump( sim_info, dataset_folder + '/' + 'sim_info.pkl')

    # Save info relevant to all datasets
    #recovar.utils.pickle_dump( noise_levels, output_folder + '/' + 'noise_levels.pkl')

if __name__ == '__main__':
    main()
    print("Done")
