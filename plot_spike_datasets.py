import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import sys
import scipy

def main():

    n_images = ["1k","10k","100k"]
    linestyles = ['dashed', 'dashdot', 'solid']
    plt.figure(1,figsize=(10, 6))
    plt.figure(2,figsize=(10, 6))
    plt.figure(3,figsize=(10, 6))
    plt.figure(4,figsize=(10, 6))

    plot_folder = f"/mnt/home/levans/ceph/spike/"
    #mpl.rcParams["linewidth"]=2
    for idx, val in enumerate(n_images): 
        output_folder = f"/mnt/home/levans/ceph/spike/recovar_experiments_{val}"

        file = open(output_folder + '/' + 'noise_levels.pkl','rb')
        noise_levels = pickle.load(file)
        file.close()  

   
        # Load in stats
        file = open(output_folder + '/' + 'pops_errors.pkl','rb')
        pops_errors = pickle.load(file)
        error_observed = pops_errors["error_observed"]
        error_predicted = pops_errors["error_predicted"]
        deconvolve_pop = pops_errors["deconvolve_pop"]
        observed_pop_soft = pops_errors["observed_pop_soft"]
        observed_pop = pops_errors["observed_pop"]
        file.close()

        file = open(output_folder + '/' + 'extra_stats.pkl','rb')
        extra_stats = pickle.load(file) 
        deconvolve_observed = extra_stats["deconvolve_observed"]
        reweight_observed = extra_stats["reweight_observed"]
        bayes_observed = extra_stats["bayes_observed"]
        reweight_pop = extra_stats["reweight_pop"]
        file.close()

        # Make a plot each time.
        plt.figure(1)
        #plt.semilogx(noise_levels, error_predicted, label=f'Analytical_{n_images}', marker='o', markersize=6, linewidth=2)
        #plt.semilogx(noise_levels, error_observed, label=f'Observed_{n_images}',  marker='s', markersize=6, linewidth=2)
        #plt.semilogx(noise_levels, deconvolve_observed, label=f'Deconvolve_{n_images}', marker='s', markersize=6, linewidth=2)
        #plt.semilogx(noise_levels, reweight_observed, label=f'Reweight_{n_images}', marker='s', markersize=6, linewidth=2)
        #plt.semilogx(noise_levels, bayes_observed, label=f'Bayes_{n_images}', marker='s', markersize=6, linewidth=2)
        plt.semilogx(noise_levels, error_predicted, label='Hard Assign, Analytical', marker='o', color='k', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, error_observed, label='Hard Assign',  marker='s', color='blue', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, deconvolve_observed, label='Deconvolve', marker='s', color='green', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, reweight_observed, label='Ensemble Reweight', marker='s', color="purple", markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, bayes_observed, label='Bayes Optimal', marker='s', color="orange", markersize=6, linewidth=2, linestyle=linestyles[idx])
        
        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('Misclassification Rate', fontsize=14)
        plt.title('Misclassification Rate vs. Noise Level', fontsize=16)
        
        if idx ==0: 
            plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_folder + '/' + 'all_curves.png')

        plt.figure(2)
        plt.semilogx(noise_levels, observed_pop[:, 0], label='Hard Assign', color='blue', marker='o', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, observed_pop_soft[:, 0], label='Soft Assign', color='orange', marker='o', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, deconvolve_pop[:, 0], label='Deconvolve', color='green', marker='s', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, reweight_pop[:, 0], label='Ensemble Reweight', color='purple', marker='s', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k", linewidth=3.0)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('% Population in state 1', fontsize=14)
        if idx ==0: 
            plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.title("Estimated Population in State 1", fontsize=14)
        plt.savefig(plot_folder + '/' + 'all_populations.png')

        # Make a plot each time.
        plt.figure(3)
        plt.semilogx(noise_levels, error_observed, label='Hard Assign',  marker='s', color='blue', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, deconvolve_observed, label='Deconvolve', marker='s', color='green', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, reweight_observed, label='Ensemble Reweight', marker='s', color="purple", markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, bayes_observed, label='Bayes Optimal', marker='s', color="orange", markersize=6, linewidth=2, linestyle=linestyles[idx])

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('Misclassification Rate', fontsize=14)
        plt.title('Misclassification Rate vs. Noise Level', fontsize=16)
 
        if idx ==0: 
            plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_folder + '/' + 'all_curves_alt.png')

        plt.figure(4)
        plt.semilogx(noise_levels, observed_pop[:, 0], label='Hard Assign', color='blue', marker='o', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, deconvolve_pop[:, 0], label='Deconvolve', color='green', marker='s', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.semilogx(noise_levels, reweight_pop[:, 0], label='Ensemble Reweight', color='purple', marker='s', markersize=6, linewidth=2, linestyle=linestyles[idx])
        plt.hlines(y=0.8, xmin=noise_levels[0], xmax=noise_levels[-1], label="True % Population", linestyle="--", color="k", linewidth=3.0)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('% Population in state 1', fontsize=14)
        if idx ==0: 
            plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.title("Estimated Population in State 1", fontsize=14)
        plt.savefig(plot_folder + '/' + 'all_populations_alt.png')




    plt.show()

if __name__ == '__main__':
    main()
    print("Done")
