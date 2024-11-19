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
import pickle
import argparse
import sys
reload(simulator)
import scipy
import matplotlib.pyplot as plt

# added in
import cvxpy as cp
import seaborn as sns
sns.set_style("ticks")
sns.color_palette("colorblind")

def confusion_and_deconvolve(assignments, true_assignments, error_predicted):
    # Compute the gamma from the note.
    confus = confusion_matrix(assignments, true_assignments)
    if confus.size > 1:
        error_observed = (confus[1,0] + confus[0,1] ) / assignments.size
    else:
        error_observed = 0
        
    # Apply deconvolution on the labels
    observed_pop, deconvolve_pop, deconvolve_matrix = deconvolve_assignments(assignments, error_predicted)
    return  error_observed, observed_pop, deconvolve_pop, deconvolve_matrix

def deconvolve_assignments(assignments, error_predicted):
    observed_pop = np.array([np.mean(assignments==0), np.mean(assignments==1)])
    deconvolve_matrix = np.array( [ [1- error_predicted, error_predicted], [error_predicted, 1- error_predicted] ])
    
    # Run constrained optimization 
    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(deconvolve_matrix @ x - observed_pop)) 
    constraints = [0 <= x, x<= 1, sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    deconvolve_pop = x.value 
    return observed_pop, deconvolve_pop, deconvolve_matrix

def deconvolve_assignments_alt(assignments, error_predicted):
    # Previous code used for this
    observed_pop = np.array([np.mean(assignments==0), np.mean(assignments==1)])
    deconvolve_matrix = np.array( [ [1- error_predicted, error_predicted], [error_predicted, 1- error_predicted] ])
    deconvolve_pop = np.linalg.solve(deconvolve_matrix, observed_pop)  
    return observed_pop, deconvolve_pop, deconvolve_matrix

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", default="/mnt/home/levans/ceph/spike/recovar_experiments_10k_new", type=str)

    args = parser.parse_args()
    output_folder = args.output_folder

    disc_type_infer = 'cubic'
    grid_size = 256
 
    # Load in things
    file = open(output_folder + '/' + 'noise_levels.pkl','rb')
    noise_levels = pickle.load(file)

    error_observed = np.zeros(noise_levels.size)
    error_predicted= np.zeros(noise_levels.size)
    deconvolve_pop = np.zeros((noise_levels.size, 2))
    deconvolve_pop_alt = np.zeros((noise_levels.size, 2))
    observed_pop = np.zeros((noise_levels.size, 2))
    observed_pop_soft = np.zeros((noise_levels.size, 2))

    for idx, noise_level in enumerate(noise_levels):

        dataset_folder = output_folder + '/' + f'dataset{idx}/'
        print(f"Starting at noise level {idx} of {len(noise_levels)}") 
        
        # Load in simulation data
        file = open(dataset_folder + '/' + 'sim_info.pkl','rb')
        sim_info = pickle.load(file)
        file.close()  

        # Load datasets and volumes
        # Volumes are scaled so that images are normalized. So they have a slightly different scale for each dataset.
        print(sim_info['volumes_path_root']) 
        volumes = simulator.load_volumes_from_folder(sim_info['volumes_path_root'], sim_info['grid_size'] , sim_info['trailing_zero_format_in_vol_name'], normalize=False )
        gt_volumes = volumes * sim_info['scale_vol']
        
        dataset_options = dataset.get_default_dataset_option()
        dataset_options['particles_file'] = dataset_folder + f'particles.{grid_size}.mrcs'
        dataset_options['ctf_file'] = dataset_folder + f'ctf.pkl'
        dataset_options['poses_file'] = dataset_folder + f'poses.pkl'
        cryo = dataset.load_dataset_from_dict(dataset_options, lazy = False)
        
        # Compute likelihoods
        batch_size = 100
        image_cov_noise = np.asarray(noise.make_radial_noise(sim_info['noise_variance'], cryo.image_shape))

        # transforming image assignment to log likelihoods
        # NOTE: previous code I was using, for the weird plot, used this, without the 1/2 
        #log_likelihoods = -1*image_assignment.compute_image_assignment(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer).T
        log_likelihoods = -0.5*image_assignment.compute_image_assignment(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer).T

        # Compute hard assignments, hard assignment uncertainties
        true_assignments = sim_info['image_assignment']
        hard_assignments = jnp.argmax(log_likelihoods, axis = 1)

        # Compute soft assignments and observed populations
        log_likelihood_per_image = scipy.special.logsumexp(log_likelihoods, axis=1)
        log_posteriors = log_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        observed_pop_soft[idx, :] = np.exp(scipy.special.logsumexp(log_posteriors, axis=0))
        observed_pop_soft[idx, :] /= observed_pop_soft[idx, :].sum()

        error_predicted[idx] = image_assignment.estimate_false_positive_rate(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer)

        error_observed[idx], observed_pop[idx, :], deconvolve_pop[idx, :], deconvolve_matrix = confusion_and_deconvolve(hard_assignments, true_assignments, error_predicted[idx])
        _, deconvolve_pop_alt[idx], _ = deconvolve_assignments_alt(hard_assignments, error_predicted[idx])

        print('o', error_observed[idx])
        print('p', error_predicted[idx])
        print('pops', deconvolve_pop[idx, :])
        
        print('Observed pop:', observed_pop[idx])
        print('Deconvolve mat:', deconvolve_matrix)
        print('Deconvolved pop:', deconvolve_pop[idx, :])

        # Dump results to file
        likelihoods_assignments = { 'log_likelihoods': log_likelihoods, 
                  'hard_assignments' : hard_assignments,
                  'true_assignments' : sim_info['image_assignment'], 
                  } 
        recovar.utils.pickle_dump(likelihoods_assignments, dataset_folder + '/' + 'likelihoods_assignments.pkl')
        recovar.utils.pickle_dump({'error_observed' : error_observed, \
                                    'error_predicted' : error_predicted, \
                                    'deconvolve_pop' : deconvolve_pop, \
                                    'observed_pop_soft': observed_pop_soft, \
                                    'observed_pop' : observed_pop, \
                                    'deconvolve_pop_alt':deconvolve_pop_alt}, \
                                output_folder + '/' + 'pops_errors.pkl') 
        # Make a plot each time.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogx(noise_levels[:idx+1], error_predicted[:idx+1], label='Analytical', color='blue', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], error_observed[:idx+1], label='Observed', color='green', marker='s', markersize=6, linewidth=2)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('False Positive Rate', fontsize=14)
        plt.title('False Positive Rate vs. Noise Level', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder + '/' + 'curve.png')

        plt.figure(figsize=(10, 6))
        plt.semilogx(noise_levels[:idx+1], observed_pop[:idx+1, 0], label='Hard Assign', color='blue', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], observed_pop_soft[:idx+1, 0], label='Soft Assign', color='orange', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], deconvolve_pop[:idx+1, 0], label='Deconvolve', color='green', marker='s', markersize=6, linewidth=2)
        plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k", linewidth=3.0)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('% Population in state 1', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder + '/' + 'populations.png')

if __name__ == '__main__':
    main()
    print("Done")