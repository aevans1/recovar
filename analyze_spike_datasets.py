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

def normalize_weights(log_weights):
    weighted_alphas = np.exp(log_weights)
    weighted_alphas = weighted_alphas / np.sum(weighted_alphas)
    return weighted_alphas

def expectation_maximization_weights(log_Pij, log_weights_init=None, num_iterations=None):
    """
    This function updates the weights according to the expectation maximization
     algorithm for mixture models.
    """
    num_images = log_Pij.shape[0]
    if log_weights_init is None:
        log_weights = np.zeros((1, log_Pij.shape[1]))
        log_weights = np.log(np.array([[0.9, 0.1]]))
    else:
        log_weights = log_weights_init

    log_weights = np.log(normalize_weights(log_weights))

    norms = np.zeros(num_iterations)
    loss = np.zeros(num_iterations)
    for k in range(num_iterations):
        log_likelihood_per_image = scipy.special.logsumexp(log_Pij + log_weights, axis=1)
        log_weighted_likelihoods = log_Pij + log_weights
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_weights = scipy.special.logsumexp(log_posteriors - np.log(num_images), axis=0)

        # compute two parameters to monitor for convergence: norm of weights, and log marginal likelihood
        norms[k] = np.linalg.norm(normalize_weights(log_weights))
        loss[k] = -1*(1/num_images)*np.sum(log_likelihood_per_image)
    log_weights = np.log(normalize_weights(log_weights))
    return log_weights, norms, loss

def classify_with_prior(pop, log_likelihoods, true_assignments):

    log_prior =  np.log(np.expand_dims(pop, axis=1)).T
    assignments = jnp.argmax(log_likelihoods + log_prior , axis = 1)

    confus = confusion_matrix(assignments, true_assignments)
    if confus.size > 1:
        error_observed = (confus[1,0] + confus[0,1] ) / assignments.size
    else:
        error_observed = 0
    return error_observed


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", default="/mnt/home/levans/ceph/spike/recovar_experiments_100k", type=str)

    args = parser.parse_args()
    output_folder = args.output_folder

    # Load in things
    file = open(output_folder + '/' + 'noise_levels.pkl','rb')
    noise_levels = pickle.load(file)
    file.close()  

    print("noiselevellsss!!")
    print(noise_levels)

    file = open(output_folder + '/' + 'pops_errors.pkl','rb')
    pops_errors = pickle.load(file)
    error_observed = pops_errors["error_observed"]
    error_predicted = pops_errors["error_predicted"]
    deconvolve_pop = pops_errors["deconvolve_pop"]
    deconvolve_pop_alt = pops_errors["deconvolve_pop_alt"]
    observed_pop_soft = pops_errors["observed_pop_soft"]
    observed_pop = pops_errors["observed_pop"]
    file.close()  

    volume_distribution = np.array([0.8, 0.2])
   
    reweight_pop = np.zeros((len(noise_levels), 2))
    bayes_observed = np.zeros(len(noise_levels))
    deconvolve_observed = np.zeros(len(noise_levels))
    reweight_observed = np.zeros(len(noise_levels))

    for idx, noise_level in enumerate(noise_levels):

        dataset_folder = output_folder + '/' + f'dataset{idx}/'
        print(f"Starting at noise level {idx} of {len(noise_levels)}") 

        # Load in stats
        file = open(dataset_folder + '/' + 'likelihoods_assignments.pkl','rb')
        likelihoods_assignments = pickle.load(file)
        file.close()  

        log_likelihoods = likelihoods_assignments['log_likelihoods']
        hard_assignments = likelihoods_assignments['hard_assignments']
        true_assignments = likelihoods_assignments['true_assignments']

        # Get likelihood weights
        log_weights, norms, loss = expectation_maximization_weights(log_likelihoods, num_iterations=5000)  
        reweight_pop[idx, :] = np.exp(log_weights)
        print('cryoER weights:', reweight_pop[idx, :])
        print('deconv weights:', deconvolve_pop[idx, :])

        deconvolve_observed[idx] =  classify_with_prior(deconvolve_pop[idx, :], log_likelihoods, true_assignments)
        reweight_observed[idx] =  classify_with_prior(reweight_pop[idx, :], log_likelihoods, true_assignments)
        bayes_observed[idx] =  classify_with_prior(volume_distribution, log_likelihoods, true_assignments)

        # Dump results to file
        extra_stats = {'deconvolve_observed' : deconvolve_observed, \
                       'reweight_observed' : reweight_observed, \
                       'bayes_observed' : bayes_observed, \
                       'reweight_pop' : reweight_pop, \
                        } 
        recovar.utils.pickle_dump(extra_stats, output_folder + '/' + 'extra_stats.pkl')
        
        # Dump results to file
        expec_maxim_stats = {'loss' : loss, \
                       'norms' : norms} 
        recovar.utils.pickle_dump(expec_maxim_stats, dataset_folder + '/' + 'expec_maxim_stats.pkl')

        # Make a plot each time.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogx(noise_levels[:idx+1], error_predicted[:idx+1], label='Hard Assign, Analytical', color='k', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], error_observed[:idx+1], label='Hard Assign', color='blue', marker='s', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], deconvolve_observed[:idx+1], label='Deconvolve', color='green', marker='s', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], reweight_observed[:idx+1], label='Reweight', color='purple', marker='s', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], bayes_observed[:idx+1], label='Bayes Optimal', color='orange', marker='s', markersize=6, linewidth=2)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('Misclassification Rate', fontsize=14)
        plt.title('Misclassification Rate vs. Noise Level', fontsize=16)
 
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder + '/' + 'curve_extra.png')

        plt.figure(figsize=(10, 6))
        plt.semilogx(noise_levels[:idx+1], observed_pop[:idx+1, 0], label='Hard Assign', color='blue', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], observed_pop_soft[:idx+1, 0], label='Soft Assign', color='orange', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], deconvolve_pop[:idx+1, 0], label='Deconvolve', color='green', marker='s', markersize=6, linewidth=2)
        #plt.semilogx(noise_levels[:idx+1], deconvolve_pop_alt[:idx+1, 0], label='Deconvolve_alt', color='red', marker='s', markersize=6, linewidth=2)
        plt.semilogx(noise_levels[:idx+1], reweight_pop[:idx+1, 0], label='Ensemble Reweight', color='purple', marker='s', markersize=6, linewidth=2)
        plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k", linewidth=3.0)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('% Population in state 1', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder + '/' + 'populations_extra.png')

    plt.show()

if __name__ == '__main__':
    main()
    print("Done")
