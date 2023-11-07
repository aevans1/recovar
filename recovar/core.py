import jax
import jax.numpy as jnp
import numpy as np
import functools
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jax.numpy)

## Low level functions for cryo-EM. Handling of indexing, slicing, adjoint of slicing, CTF, etc.

@functools.partial(jax.jit, static_argnums=[1])
def vol_indices_to_vec_indices(vol_indices, vol_shape):
    return jnp.ravel_multi_index(vol_indices.T, vol_shape, mode = 'clip')

def vec_indices_to_vol_indices(vec_indices, vol_shape):
    vol_indices = jnp.unravel_index(vec_indices, vol_shape)
    vol_indices = jnp.stack(vol_indices, axis = -1)
    return vol_indices

def vol_indices_to_frequencies(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid =  (vol_shape//2 +  (vol_shape%2 ==1) * 1).astype(int)
    return vol_indices - mid_grid

def frequencies_to_vol_indices(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid =  (vol_shape//2 +  (vol_shape%2 ==1) * 1).astype(int)
    return vol_indices + mid_grid

def vec_indices_to_frequencies(vec_indices, vol_shape):
    return vol_indices_to_frequencies(vec_indices_to_vol_indices(vec_indices, vol_shape), vol_shape)

def frequencies_to_vec_indices(frequencies, vol_shape):
    return vol_indices_to_vec_indices(frequencies_to_vol_indices(frequencies, vol_shape), vol_shape)

@jax.jit
def P_mat_vec(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid] 

# Used to project the mean
batch_P_mat_vec = jax.vmap(P_mat_vec, (None, 0))

# Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
@functools.partial(jax.jit, static_argnums=0)
def sum_batch_P_adjoint_mat_vec(volume_size, image_vecs, plane_indices_on_grids):
    volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
    volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
    return volume_vec

nosum_batch_P_adjoint_mat_vec = jax.vmap( sum_batch_P_adjoint_mat_vec, in_axes = (None, 0,0)) 

@functools.partial(jax.jit, static_argnums=0)
def sum_adj_forward_model(volume_size, images, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return sum_batch_P_adjoint_mat_vec(volume_size, images * jnp.conj(CTF_val_on_grid_stacked), plane_indices_on_grid_stacked)


@jax.jit
def forward_model(volume_vec, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return batch_P_mat_vec(volume_vec, plane_indices_on_grid_stacked) * CTF_val_on_grid_stacked


@jax.jit
def translate_single_image(image, translation, lattice):
    # lattice = cu.get_unrotated_plane_grid_points(image_shape)
    phase_shift = jnp.exp( 1j * -2 * jnp.pi * (lattice @ translation[:,None] ) )[...,0]
    return image.reshape(-1) * phase_shift

batch_translate = jax.vmap(translate_single_image, in_axes = (0,0, None))

@functools.partial(jax.jit, static_argnums=2)
def translate_images(image, translation, image_shape):
    twod_lattice = get_unrotated_plane_coords(image_shape, voxel_size =1, scaled = True )[:,:2]
    return batch_translate(image, translation, twod_lattice).astype(image.dtype)


def get_unrotated_plane_grid_points(image_shape):
    unrotated_plane_indices =  ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = 1, scaled = False)
    unrotated_plane_indices = jnp.concatenate( [unrotated_plane_indices, jnp.zeros(unrotated_plane_indices.shape[0], dtype = unrotated_plane_indices.dtype )[...,None] ], axis = -1)
    return unrotated_plane_indices

def get_unrotated_plane_coords(image_shape, voxel_size, scaled =True ):
    # These are scaled with voxel size.
    plane_coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = voxel_size, scaled = scaled)
    plane_coords = jnp.concatenate( [plane_coords, jnp.zeros(plane_coords.shape[0], dtype = plane_coords.dtype)[...,None] ], axis = -1)
    return plane_coords

## JITTING THIS MAY CAUSE WEIRD ERRORS...
# @functools.partial(jax.jit, static_argnums=[1,2,3])
def get_nearest_gridpoint_indices(rotation_matrix, image_shape, volume_shape, grid_size):
    rotated_plane = get_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size)
    rotated_indices = round_to_int(rotated_plane)
    rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_indices


@functools.partial(jax.jit, static_argnums=[1,2,3])
def get_gridpoint_coords(rotation_matrix, image_shape, volume_shape, grid_size):
    unrotated_plane_indices = get_unrotated_plane_grid_points(image_shape)

    rotated_plane = jnp.matmul(unrotated_plane_indices, rotation_matrix, precision = jax.lax.Precision.HIGHEST)
    rotated_coords = rotated_plane + jnp.floor(1.0 * grid_size/2)
    
    # rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_coords

@jax.jit
def round_to_int(array):
    return jax.lax.round(array).astype(jnp.int32)


batch_get_nearest_gridpoint_indices = jax.vmap(get_nearest_gridpoint_indices, in_axes =(0, None, None, None) ) 
batch_get_gridpoint_coords = jax.vmap(get_gridpoint_coords, in_axes =(0, None, None, None) ) 

@functools.partial(jax.jit, static_argnums=[1,2,3])
def get_rotated_plane_coords(rotation_matrix, image_shape, voxel_size, scaled = True):
    unrotated_plane_indices = get_unrotated_plane_coords(image_shape, voxel_size, scaled = scaled)
    rotated_plane = unrotated_plane_indices @ rotation_matrix
    return rotated_plane

batch_get_rotated_plane_coords = jax.vmap(get_rotated_plane_coords, in_axes = (0, None, None, None))



# @functools.partial(jax.jit, static_argnums = [4,5,6,7,8])    
def get_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type):    
    order = 1 if disc_type == "linear_interp" else 0
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, order)
    
    
## This is about twice faster. It doesn't do any checking, I guess? 
def get_slices_nomap(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type):
    grid_point_indices = batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
    return volume[grid_point_indices] 



    
@functools.partial(jax.jit, static_argnums = [2,3,4,5])    
def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, order):
    # import pdb; pdb.set_trace()
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T
    slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
    return slices


# This can only run on CPU.
def get_nufft_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size ):
    plane_coords_mol = batch_get_rotated_plane_coords(rotation_matrices, image_shape, voxel_size, True) 
    clean_image_mol = compute_volume_projections_with_nufft(volume, plane_coords_mol, voxel_size)
    return clean_image_mol

def compute_volume_projections_with_nufft(volume, plane_coords, voxel_size):
    # This is here because I don't want to impose the dependencies for nufft. If you want to run this, you should 
    # pip install finufft
    import generate_synthetic_molecule as gsm
    # plane_coords = cu.get_unrotated_plane_coords(image_shape, voxel_size, scaled =True )
    plane_coords_vec = np.array(plane_coords.reshape(-1, 3)).astype(np.float64)
    X_ims = gsm.get_fourier_transform_of_volume_at_freq_coords(np.array(volume).astype(np.complex128), plane_coords_vec, voxel_size )
    X_ims = X_ims.reshape(plane_coords.shape[:-1])
    return X_ims
    

## CTF functions.

@jax.jit
def compute_ctf_crydgrn(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor):
    '''
    Adapted for JAX from cryoDRGN: https://github.com/zhonge/cryodrgn
    
    
    Compute the 2D CTF
   
    Input: 
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees 
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * jnp.pi / 180
    phase_shift = phase_shift * jnp.pi / 180
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5
    x = freqs[...,0]
    y = freqs[...,1]
    ang = jnp.arctan2(y,x)
    s2 = x**2 + y**2
    df = .5*(dfu + dfv + (dfu-dfv)*jnp.cos(2*(ang-dfang)))
    gamma = 2*jnp.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    ctf = (1-w**2)**.5*jnp.sin(gamma) - w*jnp.cos(gamma) 
    if bfactor is not None:
        ctf *= jnp.exp(-bfactor/4*s2)
    return ctf

@jax.jit
def compute_ctf_crydgrn_packed(freqs, CTF):
    return compute_ctf_crydgrn(freqs, CTF[0], CTF[1], CTF[2], CTF[3], CTF[4], CTF[5], CTF[6], CTF[7]) * CTF[8]

batch_compute_ctf_crydgrn = jax.vmap(compute_ctf_crydgrn_packed, in_axes = (None, 0))


def compute_ctf_cryodgrn_wrapper(CTF_params, image_shape, voxel_size):
    psi = get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2]
    return batch_compute_ctf_crydgrn(psi, CTF_params)


