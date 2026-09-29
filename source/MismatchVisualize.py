import os.path
from functools import lru_cache

import numpy as np
import torch
from typing import Union, Tuple, List
# from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
import nibabel as nib
from scipy.ndimage import zoom

@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map.numpy()

def visualize(shape=[240,240,155],data_path="/home/cyx/Codes/SAM_MED3D_for_CLS/MMAN_Feature/noText",layer=0,attn_map=False):

    save_matrix = np.zeros(shape)
    record_matrix = np.zeros(shape)
    
    # times_matrix = np.ones([128,128,128])
    times_matrix = compute_gaussian(tile_size=(128,128,128))
    
    record_matrix[:128, :128, :128] += times_matrix
    record_matrix[:128, 112:240, :128] += times_matrix
    record_matrix[112:240, :128, :128] += times_matrix
    record_matrix[112:240, 112:240, :128] += times_matrix
    record_matrix[:128, :128, 27:155] += times_matrix
    record_matrix[:128, 112:240, 27:155] += times_matrix
    record_matrix[112:240, :128, 27:155] += times_matrix
    record_matrix[112:240, 112:240, 27:155] += times_matrix
    

    importance_map = compute_gaussian(tile_size=(128,128,128))

    data_list = []

    for i in range(8):
        data_np_path = os.path.join(data_path,f"{i}_{layer}.npy")
        numpy_data = np.load(data_np_path)
        if attn_map:
            to_data = np.mean(numpy_data, axis=0)
            max_map = np.mean(to_data, axis=0)

            if layer in [2, 5, 8, 11]:
                temp_map = max_map.reshape(8, 8, 8)
            else:
                temp_map = max_map.reshape(14, 14, 14)[:8, :8, :8]  # 8*8*8
            zoomed_array = zoom(temp_map, zoom=16, order=3)

        else:
            numpy_data = np.mean(numpy_data,axis=1,keepdims=True)
            numpy_data = numpy_data[0][0]
            d = numpy_data.shape[-1]
            zoom_factors = [128/d]*3
            zoom_factors = tuple(zoom_factors)
            zoomed_array = zoom(numpy_data,zoom_factors,order=3)

        weighted_data = zoomed_array * importance_map
        data_list.append(weighted_data)
        print(f"Processed {i}")


    save_matrix[:128, :128, :128] += data_list[0]
    save_matrix[:128, 112:240, :128] += data_list[1]
    save_matrix[112:240, :128, :128] += data_list[2]
    save_matrix[112:240, 112:240, :128] += data_list[3]
    save_matrix[:128, :128, 27:155] += data_list[4]
    save_matrix[:128, 112:240, 27:155] += data_list[5]
    save_matrix[112:240, :128, 27:155] += data_list[6]
    save_matrix[112:240, 112:240, 27:155] += data_list[7]

    final_map = save_matrix/record_matrix

    return final_map

if __name__ == "__main__":
    data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/MMAN_Feature/noText"
    name= data_path.split("/")[-1]
    layer = 0
    nii_gz = nib.load("/home/cyx/Datasets/BraTS2021_01731_t2.nii.gz")

    affine = nii_gz.affine
    header = nii_gz.header

    # for layer in range(1,4):
    feature_map = visualize(layer=layer,data_path=data_path,attn_map=True)
    new_img = nib.Nifti1Image(feature_map,affine,header)
    nib.save(new_img,f"{name}_layer{layer}.nii.gz")



