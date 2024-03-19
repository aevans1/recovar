from importlib import reload
import recovar.config
import numpy as np
from recovar import dataset
import jax.numpy as jnp
from recovar import locres
from recovar import adaptive_kernel_discretization
import logging
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
logger = logging.getLogger(__name__)

def pick_minimum_discretization_size(ndim, log_likelihoods, q = 0.5, min_images = 50  ):
    if ndim >0:
        disc_latent_dist = recovar.latent_density.get_log_likelihood_threshold(k = ndim, q=0.5)
    else:
        disc_latent_dist = -1
    return np.max( [ np.sort(log_likelihoods)[min_images], disc_latent_dist] )

# def pick_heterogeneity_bins(ndim, log_likelihoods, q = 0.5, min_images = 50, n_bins = 11):
#     disc_latent_dist = pick_minimum_discretization_size(ndim, log_likelihoods, q , min_images )
#     max_latent_dist = np.percentile(log_likelihoods, 0.9)
        
    return (np.linspace(1, 6 , n_bins ) **2) * disc_latent_dist

def pick_heterogeneity_bins2(ndim, log_likelihoods, q = 0.5, min_images = 50, n_bins = 11):
    disc_latent_dist = pick_minimum_discretization_size(ndim, log_likelihoods, q , min_images )

    max_latent_dist = np.percentile(log_likelihoods, 90)
    # dist = np.linspace(np.sqrt(disc_latent_dist), np.sqrt(max_latent_dist), 11 ) **2
    return np.linspace(np.sqrt(disc_latent_dist), np.sqrt(max_latent_dist), n_bins ) **2


def make_volumes_kernel_estimate_from_results(latent_point, results, ndim, cryos = None, n_bins = 11, output_folder = None, B_factor = 0, metric_used = "locmost_likely", n_min_images = 50 ):

    cryos = dataset.load_dataset_from_args(results['input_args'], lazy = False) if cryos is None else cryos
    output_folder = results['input_args'].outdir + "/output/" if output_folder is None else output_folder
    print("Dumping to ", output_folder)
    recovar.output.mkdir_safe(output_folder)
    noise_variance = results['cov_noise']
    latent_points = latent_point[None]

    log_likelihoods = recovar.latent_density.compute_latent_quadratic_forms_in_batch(latent_points[:,:ndim], results['zs'][ndim], results['cov_zs'][ndim])[...,0]
    heterogeneity_distances = [ log_likelihoods[:cryos[0].n_images], log_likelihoods[cryos[0].n_images:] ]
    if metric_used == "global":
        a = 1
        # make_volumes_kernel_estimate( heterogeneity_distances, cryos, noise_variance, output_folder, ndim, n_bins) 
    else:
        print("CHOOSING THREHSOLD ONLY BASED ON NUMBER OF IMAGES! FIX?")
        make_volumes_kernel_estimate_local(heterogeneity_distances, cryos, noise_variance, output_folder, -1, n_bins, B_factor, tau = None, n_min_images = n_min_images, metric_used = metric_used)
    
from recovar import relion_functions


# def make_volumes_kernel_estimate(heterogeneity_distances, cryos, noise_variance, output_folder, ndim, bins, tau = None, n_min_images = 80):

#     if type(bins) == int:
#         heterogeneity_bins = pick_heterogeneity_bins2(ndim, heterogeneity_distances[1], 0.5, n_min_images, n_bins = bins)
#     else:
#         heterogeneity_bins = bins

#     residual_threshold = heterogeneity_bins[0]

#     n_images_per_bin = [ np.sum(heterogeneity_distances[1] <= b) for b in heterogeneity_bins ]
#     print(n_images_per_bin)
#     # import pdb; pdb.set_trace()
#     estimates = [None, None]
#     lhs = [None, None]
#     rhs = [None, None]
#     for k in range(2):
#         estimates[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, compute_lhs_rhs=False)

#     cross_validation_estimators = [None, None]
#     for k in range(2):
#         cross_validation_estimators[k], lhs[k], rhs[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins[0:1], tau= tau, compute_lhs_rhs=True)

    
#     import recovar
#     recovar.utils.pickle_dump( { "lhs" : lhs, "rhs" : rhs } ,  output_folder  + "lhs_rhs.pkl")

#     logger.info(f"Computing estimates done")

#     import jax.numpy as jnp
#     index_array_vol, fdisc, residuals_avged,  summed_residuals  = adaptive_kernel_discretization.pick_best_heterogeneity_from_residual((estimates[0].T)[...,None], 
#                                         cryos[1], heterogeneity_distances[1], heterogeneity_bins, 
#                                         residual_threshold = residual_threshold , min_number_of_images_in_bin = 50)
#     logger.info(f"Computing estimates done")


#     opt_halfmaps = [None, None]
#     for k in range(2):
#         opt_halfmaps[k] = jnp.take_along_axis(estimates[k].T , np.expand_dims(index_array_vol, axis=-1), axis=-1)

#     recovar.output.save_volume(opt_halfmaps[0], output_folder + "optimized_half1_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)
#     recovar.output.save_volume(opt_halfmaps[1], output_folder + "optimized_half2_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)

#     recovar.output.save_volumes(estimates[0], output_folder + "estimates_half1_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)
#     recovar.output.save_volumes(estimates[1], output_folder + "estimates_half2_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)
    
#     recovar.utils.pickle_dump( { "index_array_vol" : index_array_vol, "fdisc" : fdisc, "residuals_avged" : residuals_avged, "n_images_per_bin" :n_images_per_bin, "summed_residuals" : summed_residuals } ,  output_folder  + "stuff.pkl")
#     return estimates, opt_halfmaps, index_array_vol, fdisc, residuals_avged



def make_volumes_kernel_estimate_local(heterogeneity_distances, cryos, noise_variance, output_folder, ndim, bins, B_factor, tau = None, n_min_images = 50, metric_used = "locres"):

    if type(bins) == int:
        heterogeneity_bins = pick_heterogeneity_bins2(ndim, heterogeneity_distances[1], 0.5, n_min_images, n_bins = bins)
    else:
        heterogeneity_bins = bins

    logger.info(f"bins {heterogeneity_bins}")
    n_images_per_bin = [ np.sum(heterogeneity_distances[1] <= b) for b in heterogeneity_bins ]
    # logger.info(f"images per bin {*n_images_per_bin}")
    print("images per bin", n_images_per_bin)
    # print(n_images_per_bin)
    # import pdb; pdb.set_trace()
    estimates = [None, None]
    lhs, rhs = [None, None], [None, None]
    for k in range(2):
        estimates[k] = adaptive_kernel_discretization.less_naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, grid_correct=False)

        # estimates[k], lhs[k], rhs[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, compute_lhs_rhs=True)

        estimates[k] = ftu.get_idft3(estimates[k].reshape(-1, *cryos[0].volume_shape)).real.astype(np.float32)

    cross_validation_estimators = [None, None]
    for k in range(2):
        cross_validation_estimators[k], lhs[k], rhs[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins[0:1], tau= tau, compute_lhs_rhs=True, grid_correct=False)

        # Zero out things after Nyquist - these won't be used in CV
        lhs[k] = (lhs[k][0] * cryos[0].get_valid_frequency_indices()).reshape(cryos[0].volume_shape)

        cross_validation_estimators[k] = ftu.get_idft3(cross_validation_estimators[k].reshape(cryos[0].volume_shape)).real.astype(np.float32)


    logger.info(f"Computing estimates done")

    from_ft = False

    ml_choice, ml_errors = choice_most_likely(estimates[0], estimates[1], cross_validation_estimators[0], cross_validation_estimators[1], lhs[0], lhs[1], cryos[0].voxel_size)
    # choice_most_likely(estimates0, estimates1, target0, target1, noise_variances_target0, noise_variances_target1, voxel_size)
    locres_choice, locres_score, auc_choice, auc_score = choice_best_locres(estimates[0], estimates[1][0], cryos[0].voxel_size)

    # print(f"using local resolution")    
    # recovar.output.save_volumes(auc_score, output_folder + "auc", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)

    # if metric_used == "locres":
    #     choice, errors, choice2, auc_score = choice_best_locres(estimates[0], estimates[1], 0, cryos[0].voxel_size)
    #     print(f"using local resolution")    
    #     recovar.output.save_volumes(auc_score, output_folder + "auc", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)
    # elif metric_used == "locres_auc":
    #     choice2, errors, choice, auc_score = choice_best_locres(estimates[0], estimates[1], 0, cryos[0].voxel_size)
    #     print(f"using local resolution auc")    
    #     recovar.output.save_volumes(auc_score, output_folder + "auc", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)
    # elif metric_used == "locmost_likely":
    #     choice, errors = choice_most_likely(estimates[0], estimates[1], lhs[0], lhs[1], 0, cryos[0].voxel_size)
    # else:
    #     choice, errors = choice_best_local_error(estimates[0], estimates[1], 0, cryos[0].voxel_size)
    # logger.info(f"Picking best done")
    
    estimates = np.asarray(estimates)
    for k in range(2):
        for i in range(estimates[k].shape[0]):
            gridding_correct = "square"
            grid_fn = relion_functions.griddingCorrect_square if gridding_correct == "square" else relion_functions.griddingCorrect
            kernel_width = 1
            order =1
            estimates[k][i], _ = grid_fn(estimates[k][i].reshape(cryos[0].volume_shape), cryos[0].grid_size, cryos[0].volume_upsampling_factor/kernel_width, order = order)

            # estimates[i] = relion_functions.griddingCorrect

    def use_choice_and_filter(choice, name):

        opt_halfmaps = [None, None]
        for k in range(2):
            opt_halfmaps[k] = jnp.take_along_axis(estimates[k] , choice[None], axis=0)[0]

        best_filtered, best_filtered_res, best_auc, fscs, resols = locres.local_resolution(opt_halfmaps[0], opt_halfmaps[1], B_factor, cryos[0].voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = False)

        recovar.utils.write_mrc(output_folder + name + "optimized_locres_filtered.mrc", best_filtered, voxel_size = cryos[0].voxel_size)
        recovar.utils.write_mrc(output_folder + name + "optimized_locres.mrc", best_filtered_res, voxel_size = cryos[0].voxel_size)
        recovar.utils.write_mrc(output_folder + name +"optimized_auc.mrc", best_auc, voxel_size = cryos[0].voxel_size)
        recovar.utils.write_mrc(output_folder + name + "optimized_choice.mrc", choice, voxel_size = cryos[0].voxel_size)

        output_dict = { "heterogeneity_bins" : heterogeneity_bins, "n_images_per_bin" :n_images_per_bin, "fscs" : fscs }
        recovar.utils.pickle_dump(output_dict ,  output_folder + name + "params.pkl")



    use_choice_and_filter(ml_choice, "ml_")
    use_choice_and_filter(locres_choice, "locres_")
    use_choice_and_filter(auc_choice, "auc_")


    recovar.output.save_volumes(estimates[0], output_folder + "estimates_half1_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)
    recovar.output.save_volumes(estimates[1], output_folder + "estimates_half2_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)

    # use_choice_and_filter(ml_choice, "ml_")
    # use_choice_and_filter(locres_choice, "locres_")
    # use_choice_and_filter(auc_choice, "auc_")

    # recovar.output.save_volumes(errors, output_folder + "errors", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size, from_ft = from_ft)

    # best_filtered, best_filtered_res, best_auc, fscs, resols = locres.local_resolution(opt_halfmaps[0], opt_halfmaps[1], B_factor, cryos[0].voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = False)

    # recovar.utils.write_mrc(output_folder + "optimized_locres_filtered.mrc", best_filtered, voxel_size = cryos[0].voxel_size)

    # recovar.utils.write_mrc(output_folder + "optimized_locres.mrc", best_filtered_res, voxel_size = cryos[0].voxel_size)
    # use_choice_and_filter(choice, name)

    # recovar.utils.write_mrc(output_folder + "optimized_auc.mrc", best_auc, voxel_size = cryos[0].voxel_size)

    # recovar.utils.write_mrc(output_folder + "optimized_choice.mrc", choice, voxel_size = cryos[0].voxel_size)

    # output_dict = { "heterogeneity_bins" : heterogeneity_bins, "n_images_per_bin" :n_images_per_bin, "fscs" : fscs }
    # if metric_used == "locres":
    #     output_dict["auc"] = choice2
    # # i_loc_auc, fscs, local_resols
    # recovar.utils.pickle_dump(output_dict ,  output_folder  + "stuff.pkl")
    import pdb; pdb.set_trace()
    return best_filtered


# make_volumes_kernel_estimate_from_results(latent_points[0], results, ndim, cryos = None)

def choice_most_likely(estimates0, estimates1, target0, target1, noise_variances_target0, noise_variances_target1, voxel_size):
    from recovar import locres
    reload(locres)

    n_estimators = estimates0.shape[0]
    errors = np.zeros_like(estimates0)
    for k in range(n_estimators):  
        errors[k] = locres.expensive_local_error_with_cov(target0, estimates1[k], voxel_size, noise_variances_target0.reshape(target0.shape), locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None)
        errors[k] += locres.expensive_local_error_with_cov(estimates0[k], target1, voxel_size, noise_variances_target1.reshape(target0.shape), locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None)

    choice = np.argmin(errors, axis=0)
    return choice, errors


# def choice_best_local_error(estimates0, estimates1, target_idx, voxel_size):
#     from recovar import locres
#     reload(locres)

#     n_estimators = estimates0.shape[0]
#     errors = np.zeros_like(estimates0)
#     for k in range(n_estimators):        
#         errors[k] = locres.local_error(estimates0[target_idx], estimates1[k], voxel_size)
#         errors[k] += locres.local_error(estimates1[k], estimates1[target_idx], voxel_size)

#     choice = np.argmin(errors, axis=0)
#     return choice, errors

# def choice_most_likely(estimates0, estimates1, target0, target1, noise_variances_target0, noise_variances_target1, voxel_size):
#     from recovar import locres
#     reload(locres)

#     n_estimators = estimates0.shape[0]
#     errors = np.zeros_like(estimates0)
#     for k in range(n_estimators):  
#         errors[k] = locres.expensive_local_error_with_cov(target0, estimates1[k], voxel_size, noise_variances_target0.reshape(target0.shape), locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None)
#         errors[k] += locres.expensive_local_error_with_cov(estimates0[k], target1, voxel_size, noise_variances_target1.reshape(target0.shape), locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None)

#     choice = np.argmin(errors, axis=0)
#     return choice, errors


# def choice_best_locres(estimates0, estimates1, target_idx, voxel_size):
def choice_best_locres( estimates1, target0, voxel_size):

    from recovar import locres
    reload(locres)

    n_estimators = estimates1.shape[0]
    locressol = np.zeros_like(estimates1)
    auc_score = np.zeros_like(estimates1)

    for k in range(n_estimators):
        # _, locressol[k], auc_score[k] , _, _ = locres.local_resolution(estimates0[target_idx], estimates1[k], 0, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True)
        _, _, locressol[k], auc_score[k] = locres.local_resolution(target0, estimates1[k], 0, voxel_size, locres_sampling = 15, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = False, use_v2=True)

    choice = np.argmin(locressol, axis=0)
    choice2 = np.argmax(auc_score, axis=0)

    return choice, locressol, choice2, auc_score


# def choice_best_fsc_auc(estimates0, estimates1, target_idx, voxel_size):
#     from recovar import locres
#     reload(locres)

#     n_estimators = estimates0.shape[0]
#     locressol = np.zeros_like(estimates0)
#     for k in range(n_estimators):        
#         _, _, locressol[k], _, _ = locres.local_resolution(estimates0[target_idx], estimates1[k], 0, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True)

#     choice = np.argmax(locressol, axis=0)
#     return choice, locressol


# def smooth_choice(choice, edgewidth):


#     return


#     return choice, errors


# i_fil, i_loc_res, fscs, local_resols = {}, {}, {}, {}
# fscs, local_resols = {}, {}
# fscs_gt, local_resols_gt = {}, {}
# B_factor = 100
# i_loc_res = {}
# i_loc_res_gt = {}
# i_loc_auc = {}
# i_loc_auc_gt = {}
# import jax
# voxel_size = cryo_dataset.voxel_size
# use_filter = True
# errors = {}
# for k in range(1,11,1):
#     map1 = recovar.utils.load_mrc(output_folder + "estimates_half1_unfil002.mrc")
#     map2 = recovar.utils.load_mrc(output_folder + "estimates_half2_unfil"+format(k, '03d')  +".mrc")
    
#     from recovar import locres
#     reload(locres)
#     _, i_loc_res[k], i_loc_auc[k], fscs[k], local_resols[k] = locres.local_resolution(map1, map2, B_factor, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = use_filter)

#     _, i_loc_res_gt[k], i_loc_auc_gt[k], fscs_gt[k], local_resols_gt[k] = locres.local_resolution(target_real, map2, B_factor, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = use_filter, fsc_threshold = 1/2)
#     # errors[k] = locres.local_error(map1, map2, voxel_size)
    
    