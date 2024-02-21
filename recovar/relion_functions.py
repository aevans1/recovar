import numpy as np
from recovar import core
from recovar import regularization, constants, utils

from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
ftu = fourier_transform_utils(np)

def griddingCorrect(vol_in, ori_size, padding_factor, order = 0,):

    # Correct real-space map by dividing it by the Fourier transform of the interpolator(s)
    # vol_in.setXmippOrigin()
    pixels = ftu.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) + 0.
    og_shape = vol_in.shape
    r = np.linalg.norm(pixels, axis = -1)
    vol_in = vol_in.reshape(-1)

    mask = r > 0.
    
    rval = r / (ori_size * padding_factor)
    rval[~mask] = 1.
    sinc = np.sin(np.pi * rval) / (np.pi * rval)
    sinc[~mask] = 1.

    if order ==0:
        vol_out = vol_in/ sinc
    elif order ==1:
        vol_out = vol_in/ (sinc**2)
        sinc = sinc**2
    else:
        raise ValueError("Order not implemented")
    
    return vol_out.reshape(og_shape), sinc.reshape(og_shape)

# I think this is the correct Fourier transform of the trilinear interpolator: sinc(x) * sinc(y) * sinc(z)
def griddingCorrect_square(vol_in, ori_size, padding_factor, order = 0,):
    og_shape = vol_in.shape

    pixels = ftu.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) 
    pixels_rescaled = pixels / (ori_size * padding_factor)

    def sinc(ar):
        # ar_scaled = ar / (ori_size * padding_factor)
        return jnp.where(ar == 0, 1., jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

    if order ==0:
        kernel = sinc
    elif order ==1:
        kernel = lambda x : sinc(x)**2
    else:
        raise ValueError("Order not implemented")

    kernel_ar = kernel(pixels_rescaled[:,0]) * kernel(pixels_rescaled[:,1]) * kernel(pixels_rescaled[:,2])
    # kernel_ar = kernel ** (order + 1)
    vol_out = vol_in / kernel_ar.reshape(og_shape)

    return vol_out.reshape(og_shape), kernel_ar.reshape(og_shape)


# My understanding of what relion does.
def relion_style_triangular_kernel(experiment_dataset , cov_noise,  batch_size,  disc_type = 'linear_interp', return_lhs_rhs = False ):
    
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    Ft_y, Ft_ctf = 0, 0 

    for batch, indices in data_generator:

        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        Ft_y_b, Ft_ctf_b = relion_style_triangular_kernel_batch(batch,
                                                                experiment_dataset.CTF_params[indices], 
                                                                experiment_dataset.rotation_matrices[indices], 
                                                                experiment_dataset.translations[indices], 
                                                                experiment_dataset.image_shape, 
                                                                experiment_dataset.upsampled_volume_shape, 
                                                                experiment_dataset.voxel_size, 
                                                                experiment_dataset.CTF_fun, 
                                                                disc_type, 
                                                                cov_noise)
        Ft_y += Ft_y_b
        Ft_ctf += Ft_ctf_b
    # To agree with order of other fcns.
    return Ft_ctf, Ft_y

import functools, jax
@functools.partial(jax.jit, static_argnums=[4,5,6,7,8,9])
def relion_style_triangular_kernel_batch(images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, cov_noise):
    # images = process_images(images, apply_image_mask = True)
    images = core.translate_images(images, translations, image_shape)
    Ft_y = core.adjoint_forward_model_from_map(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) / cov_noise

    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    Ft_ctf = core.adjoint_forward_model_from_map(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) / cov_noise

    return Ft_y, Ft_ctf


def adjust_regularization_relion_style(filter, volume_shape, tau = None, oversampling_factor = 1, max_res_shell = None):

    # Original code copy pasted from https://github.com/3dem/relion/blob/e5c4835894ea7db4ad4f5b0f4861b33269dbcc77/src/backprojector.cpp#L1082

    # There is an "oversampling" factor of 2 in the FSC, I guess due to the fact that they swap back and forth between a padded and unpadded grid
    if tau is not None:
        inv_tau = 1 / (oversampling_factor * tau)
        # filter_this =  jnp.where(lhs > 1e-20 , 1/ ( 0.001 * jnp.where(filter > 1e-20, filter, 0 )
        inv_tau = jnp.where( (tau < 1e-20) * (filter > 1e-20 ),  1./ ( 0.001 * filter), inv_tau)
        inv_tau = jnp.where( (tau < 1e-20) * (filter <= 1e-20 ),  0, inv_tau)

        regularized_filter = filter + inv_tau
    else:
        regularized_filter = filter

    # Take max of weight of 1/1000 of spherically averaged weight 
    # const RFLOAT weight =  XMIPP_MAX(DIRECT_A3D_ELEM(Fweight, k, i, j), DIRECT_A1D_ELEM(radavg_weight, (ires < r_max) ? ires : (r_max - 1)));
    # Compute spherically averaged 
    avged_reg = regularization.average_over_shells(regularized_filter, volume_shape, frequency_shift = 0) / 1000
    # For the things below that frequency, set them to averaged.
    if max_res_shell is not None:
        avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
    else:
        max_res_shell = volume_shape[0]//2 - 1
        # avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])

    avged_reg_volume_shape = utils.make_radial_image(avged_reg, volume_shape)

    regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume_shape)
    regularized_filter = jnp.maximum(regularized_filter, constants.EPSILON)

    return regularized_filter
