import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 12 512 512 代表不同的head DHW DHW  "/home/cyx/TEMP_Code/BraTS2021_01731/BraTS2021_01731_t2.nii.gz"
# 全部弄成max,查看情况
data_format = "/home/cyx/Codes/SAM_MED3D_for_CLS/MMAN_Feature/ErrorCase/{}_{}.npy"
temp_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/visualized_tensors/WithoutAdapterPNG/{}_T2_img.png"
multi_data = np.load("/home/cyx/Datasets/Full_data/new_valid/BraTS20_Validation_043.npy")
t2_data = multi_data[...,2]
slice_id = 73
t_2d = t2_data[:,:,slice_id].T

# 处理重叠区域的加权平均
# def blend_blocks(block1, block2, axis, overlap_size=16):
#     weights = np.linspace(0, 1, overlap_size)
#     if axis == 0:  # x 方向
#         block1[-overlap_size:] = block1[-overlap_size:] * (1 - weights[:, None, None]) + block2[:overlap_size] * weights[:, None, None]
#     elif axis == 1:  # y 方向
#         block1[:, -overlap_size:] = block1[:, -overlap_size:] * (1 - weights[None, :, None]) + block2[:, :overlap_size] * weights[None, :, None]
#     elif axis == 2:  # z 方向
#         block1[:, :, -overlap_size:] = block1[:, :, -overlap_size:] * (1 - weights[None, None, :]) + block2[:, :, :overlap_size] * weights[None, None, :]

for i in range(1):
    all_mask = np.zeros((240, 240, 155))
    ls = []

    for j in range(8):
        data = np.load(data_format.format(j,i))
        to_data = np.mean(data, axis=0)
        max_map = np.mean(to_data, axis=0)

        if i in [2, 5, 8, 11]:
            temp_map = max_map.reshape(8, 8, 8)
        else:
            temp_map = max_map.reshape(14, 14, 14)[:8, :8, :8]

        zoomed_array = zoom(temp_map, zoom=16, order=3)
        ls.append(zoomed_array)

    # 对x和y轴方向的重叠进行平滑处理
    # blend_blocks(ls[0], ls[1], axis=1)
    # blend_blocks(ls[0], ls[2], axis=0)
    # blend_blocks(ls[4], ls[5], axis=1)
    # blend_blocks(ls[4], ls[6], axis=0)
    #
    # # 对z轴方向的重叠进行平滑处理
    # blend_blocks(ls[0], ls[4], axis=2, overlap_size=101)
    # blend_blocks(ls[1], ls[5], axis=2, overlap_size=101)
    # blend_blocks(ls[2], ls[6], axis=2, overlap_size=101)
    # blend_blocks(ls[3], ls[7], axis=2, overlap_size=101)
    #
    # # 将块赋值到all_mask中
    # all_mask[:128, :128, :128] = ls[0]
    # all_mask[:128, 112:240, :128] = ls[1]
    # all_mask[112:240, :128, :128] = ls[2]
    # all_mask[112:240, 112:240, :128] = ls[3]
    # all_mask[:128, :128, 27:155] = ls[4]
    # all_mask[:128, 112:240, 27:155] = ls[5]
    # all_mask[112:240, :128, 27:155] = ls[6]
    # all_mask[112:240, 112:240, 27:155] = ls[7]

    blend = np.linspace(0, 1, 16)
    blend2 = np.linspace(0,1,16)
    blend1 = blend[None, :, None]
    blend2 = blend2[None, :]
    blend3 = blend[:, None, None]

    all_mask[:128, :128, :128] = ls[0]
    all_mask[:128, 112:240, :128] = ls[1]
    all_mask[112:240, :128, :128] = ls[2]
    all_mask[112:240, 112:240, :128] = ls[3]
    all_mask[:128, :128, 27:155] = ls[4]
    all_mask[:128, 112:240, 27:155] = ls[5]
    all_mask[112:240, :128, 27:155] = ls[6]
    all_mask[112:240, 112:240, 27:155] = ls[7]

    temp_save = all_mask[:,:,slice_id]
    temp_save = temp_save/np.max(temp_save)

    all_mask[112:240, 112:128, 27:155] = ls[6][:, -16:, :]*(1-blend1) + ls[7][:, :16, :] * blend1
    all_mask[:128, 112:128, 27:155] = ls[4][:, -16:, :] * (1 - blend1) + ls[5][:, :16, :] * blend1

    all_mask[112:128, 112:240, 27:155] = ls[5][-16:, :, :] * (1 - blend3) + ls[7][:16, :, :] * blend3
    all_mask[112:128, :128, 27:155] = ls[4][-16:, :, :] * (1 - blend3) + ls[6][:16, :, :] * blend3

    # all_mask[:128, 112:128, :128] = ls[0][:, -16:, :] * (1 - blend1) + ls[1][:, :16, :] * blend1

    # all_mask[112:240, 112:128, :128] = ls[2][:, -16:, :] * (1 - blend1) + ls[3][:, :16, :] * blend1

    # all_mask[112:128, :128, :128] = ls[0][-16:, :, :]*(1-blend3) + ls[2][:16, :, :]*blend3

    # all_mask[112:128, 112:240, :128] = ls[1][-16:, :, :] * (1 - blend3) + ls[3][:16, :, :] * blend3
    #
    # all_mask[:128, :128, 27:128] = ls[0][:, :, -101:]*(1-blend2) + ls[4][:, :, :101]*blend2
    # all_mask[112:240, :128, 27:128] = ls[2][:, :, -101:] * (1 - blend2) + ls[6][:, :, :101] * blend2
    # all_mask[:128, 112:240, 27:128] = ls[1][:, :, -101:] * (1 - blend2) + ls[5][:, :, :101] * blend2
    # all_mask[112:240, 112:240, 27:128] = ls[3][:, :, -101:] * (1 - blend2) + ls[7][:, :, :101] * blend2
    #
    # all_mask[112:128, 112:128, 27:155] /= 2

    # process overlap region



    attn_2d = all_mask[:,:,slice_id]
    attn_2d = attn_2d/np.max(attn_2d)
    # attn_2d[:,112:128] = temp_save[:,112:128]
    # attn_2d[:,108:113] = attn_2d[:,112:117]
    # attn_2d = attn_2d/np.max(attn_2d)
    # attn_2d = gaussian_filter(attn_2d,sigma=0.5)


    # plt.subplot(1,2,1)
    # plt.imshow(t_2d,cmap="gray")
    #
    # plt.subplot(1,2,2)
    plt.figure(figsize=(4,4))
    plt.imshow(t_2d,cmap="gray")
    plt.imshow(attn_2d,cmap="rainbow",alpha=0.6)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(temp_path.format(i),bbox_inches='tight',pad_inches=0)

# data = np.load("/home/cyx/Codes/SAM_MED3D_for_CLS/visualized_tensors/0_mean_resized.npy")
# multi_data = np.load("/home/cyx/Datasets/Full_data/new_valid/BraTS20_Validation_082.npy")
# t2_data = multi_data[...,2]
# t2_data = t2_data[:128,:128,:128]
# t_2d = t2_data[:,:,79].T
# attn_2d = data[:,:,79]
# #
# plt.subplot(1,2,1)
# plt.imshow(t_2d,cmap="gray")
#
# plt.subplot(1,2,2)
# plt.imshow(t_2d,cmap="gray")
# plt.imshow(attn_2d/np.max(attn_2d),cmap="rainbow",alpha=0.6)
# plt.savefig("2.png")
