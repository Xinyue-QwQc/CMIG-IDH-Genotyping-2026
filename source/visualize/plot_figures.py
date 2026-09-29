
#%%

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
feature_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/features"

feature = Path(feature_path)

for i in range(12):
    file_path = f"/home/cyx/Codes/SAM_MED3D_for_CLS/features/DW_depth{i}.npy"
    Lora_path = f"/home/cyx/Codes/SAM_MED3D_for_CLS/features/Lora_depth{i}.npy"
    data = np.load(file_path)
    lora_data = np.load(Lora_path)

    data = np.mean(data,axis=0,keepdims=False)
    lora_data = np.mean(lora_data,axis=0,keepdims=False)
    sub_data = data - lora_data
    plt.imshow(sub_data)
    plt.title(f"{i} layer")
    plt.show()
    # plt.subplot(1,2,1)
    # plt.imshow(data)
    # plt.title(f"{i} layer,dw")
    # plt.subplot(1,2,2)
    # plt.imshow(lora_data)
    # plt.title(f"{i} layer,lora")
    # plt.show()
    


# %%

#%%
