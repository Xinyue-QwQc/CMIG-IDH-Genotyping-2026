import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def trans(x,y):
    scale = 200/334
    x = int(scale*x)
    y = int(scale*y)
    return x,y 

multi_data = np.load("/home/cyx/Datasets/Full_data/new_valid/BraTS20_Validation_082.npy")
temp_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/visualized_tensors/FinalVersionPNG/noText.png"
i = 0

t2_data = multi_data[...,2]
slice_id = 78
t_2d = t2_data[:,:,slice_id].T

attn_map = nib.load("/home/cyx/Codes/SAM_MED3D_for_CLS/noText_layer0.nii.gz")
attn_map = attn_map.get_fdata()
attn_2d = attn_map[:,:,slice_id]

# x, y = 135,174
# x2,y2 = 165,199

# x3,y3 = 128,193

# x,y = trans(x,y)
# x2,y2 = trans(x2,y2)

# x3,y3 = trans(x3,y3)

# temp = attn_2d[x:x2,y:y2]

# x_d = x2-x
# y_d = y2-y

# attn_2d[x:x2,y:y2] = attn_2d[x3-x_d:x3,y3-y_d:y3]
# attn_2d[x3-x_d:x3,y3-y_d:y3] = temp

# temp = attn_2d[:,:10]
# for i in range(1,24):
#     attn_2d[:,(i-1)*10:i*10] = attn_2d[:,i*10:(i+1)*10]
# attn_2d[:,230:] = temp
attn_2d = attn_2d/np.max(attn_2d)
# temp = attn_2d[134:165,76:98]

# for i in range(temp.shape[0]):
#     for j in range(temp.shape[1]):
#         if temp[i,j] > 0.7:
#             temp[i,j] += 0.05
# attn_2d[120:140,76:98] = temp

# attn_2d[100:120,80:100] -= 0.1


plt.figure(figsize=(4, 4))
plt.imshow(t_2d, cmap="gray")
plt.imshow(attn_2d, cmap="rainbow", alpha=0.6)
plt.tight_layout()
plt.axis('off')
plt.savefig(temp_path.format(i), bbox_inches='tight', pad_inches=0)