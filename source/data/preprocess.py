import nibabel as nib 
import json 
import os 
import numpy as np 
import pickle 




def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path,data):
    with open(path,"r") as f:
        json.dump(data,f,indent=4,sort_keys=True)

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data



def load_and_process_data(modality_path,is_replace=True):

    if is_replace:
        for i in range(4):
            modality_path[str(i)] = os.path.join("/homec/kuanghl2",*modality_path[str(i)].split("/")[3:])
    
    split_path = modality_path["0"].split("/")
    
    data_path = "/"+os.path.join(*split_path[:-1],split_path[-2]+".pkl")

    if os.path.exists(data_path):
        for i in range(4):
            del modality_path[str(i)]
        modality_path["data"] = data_path
        print("Have Processed ", split_path[-2])
        return modality_path
        


    

    
    # 每一个分别是240 240 155
    try:
        images = np.stack([
            np.array(nib_load(modality_path[str(i)]),order="C",dtype='float32') for i in range(4)
        ],-1)
    except:
        print(modality_path)
        return np.random.random((240,240,155,4)),1

    # split_path = modality_path["0"].split("/")
    
    # data_path = "/"+os.path.join(*split_path[:-1],split_path[-2]+".pkl")

    for i in range(4):
        del modality_path[str(i)]

    modality_path["data"] = data_path

    mask = images.sum(-1) > 0

    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x
    
    print("Processed ", split_path[-2])
    
    with open(data_path, 'wb') as f:
        pickle.dump(images,f)


    return modality_path

new_json = read_json("/homec/kuanghl2/Codes/data_information_final.json")
del new_json["mutant_shuffle_index"]
del new_json["wild_shuffle_index"]

for i in range(len(new_json["Mutant_data"])):
    new_json["Mutant_data"][i] = load_and_process_data(new_json["Mutant_data"][i])

for i in range(len(new_json["Wild_data"])):
    new_json["Wild_data"][i] = load_and_process_data(new_json["Wild_data"][i])


write_json("/homec/kuanghl2/Codes/data_information.json",new_json)



