from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import  umap
# import seaborn as sns

def load_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

# scaler = MinMaxScaler()
scaler = StandardScaler()

# 自定义标签和符号
markers = {0: '*', 1: '^'}  # 0号标签用星星，1号标签用三角形
labels_name = {0: 'Wild Type', 1: 'Mutant Type'}  # 自定义标签名称

data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/TestSetTensor/Multimodal_MRS.npy"
data = np.load(data_path)
data = scaler.fit_transform(data)
model_name = data_path.split("/")[-1][:-4]
labels = load_json("/home/cyx/Codes/SAM_MED3D_for_CLS/test_label.json")["labels"]
labels = np.array(labels)
#,early_exaggeration=10,perplexity=20
tsne = TSNE(n_components=2,random_state=42,n_iter=2000,perplexity=22)
two_d_data = tsne.fit_transform(data)
#
# fit_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,random_state=42,metric='euclidean')
# two_d_data = fit_umap.fit_transform(data)

plt.figure(figsize=(8, 6))
for k in range(2):
    plt.scatter(
        two_d_data[labels==k,0],two_d_data[labels==k,1],
        marker = markers[k], s=100, label = labels_name[k]
    )
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.savefig(f"./TSNE_PNGS/{model_name}.png",bbox_inches='tight')