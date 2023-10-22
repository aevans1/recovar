## something
import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "result_dir",
        # dest="result_dir",
        type=os.path.abspath,
        help="result dir (output dir of pipeline)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=False,
        help="Output directory to save model",
    )


    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument(
        "--kmeans-ind",
        dest="kmeans_ind",
        type=list_of_ints,
        default=None,
        help="indices of k means centers to use as endpoints",
    )

    parser.add_argument(
        "--endpts",
        dest="endpts_file",
        default=None,
        help="end points file. It storing z values, it should be a .txt file with 2 rows, and if it is from kmeans, it should be a .pkl file (generated by analyze)",
    )

    parser.add_argument(
        "--zdim", type=int, help="Dimension of latent variable (a single int, not a list)"
    )


# args.zdim, z_st, z_end, args.outdir, args.n_vols_along_path, cryos = None, likelihood_threshold = args.q

    parser.add_argument(
        "--q", metavar=float, default=None, help="quantile used for reweighting (default = 0.95)"
    )

    parser.add_argument(
        "--n-vols", dest= "n_vols_along_path", metavar=int, default=6, help="number of volumes produced at regular interval along the path"
    )

    # parser.add_argument(
    #     "--n-std", metavar=float, default=None, help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)"
    # )

    return parser


def compute_trajectory(results, zdim, z_st, z_end, output_folder, n_vols_along_path = 6, cryos = None, likelihood_threshold = None):

    # Load dataset if not loaded
    cryos = dataset.load_dataset_from_args(results['input_args']) if cryos is None else cryos
    embedding.set_contrasts_in_cryos(cryos, results['contrasts'][zdim])


    path_folder = output_folder       
    o.mkdir_safe(path_folder)

    path_z = o.make_trajectory_plots_from_results(results, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, basis_size =zdim, compute_reproj = False, likelihood_threshold = likelihood_threshold)    

    return path_z

def compute_trajectory_from_terminal(args):
    results = o.load_results_new(args.result_dir + '/')

    if args.kmeans_ind is not None:
        kmeans_result = pickle.load(open(args.endpts_file, 'rb'))
        z_st = kmeans_result['centers'][args.kmeans_ind[0]]
        z_end = kmeans_result['centers'][args.kmeans_ind[1]]
    elif args.endpoints_file is not None:
        end_points = np.loadtxt(args.endpts_file)
        z_st = end_points[0]
        z_end = end_points[1]
    else:
        raise Exception("end point format wrong")

    compute_trajectory(results, args.zdim, z_st, z_end, args.outdir, args.n_vols_along_path, cryos = None, likelihood_threshold = args.q)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_trajectory_from_terminal(args)
