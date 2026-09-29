import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import time
import json


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
def load_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        if random.random() < 0.5:
            image = np.flip(image, 0)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        if random.random() < 0.5:
            image = np.flip(image, 2)
        return {'image': image}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']

        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 155 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        # image = image[61: 61 + 128, 61: 61 + 128, 11: 11 + 128, ...]

        return {'image': image}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']

        scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image * scale_factor + shift_factor

        return {'image': image, }


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)

        return {'image': image}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')

        return {'image': image}
    # (240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # modality num H W D
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))

        image = torch.from_numpy(image).float()

        return {'image': image}


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        # Augmentation(),
        ToTensor()
    ])
    return trans(sample)


# 同样做random flip保证得到的模型是能够对抗random flip得到的好的模型
def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        Random_Crop(),
        # Random_Flip(),
        ToTensor()
    ])

    return trans(sample)


def transform_test(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        Random_Crop(),
        ToTensor()
    ])

    return trans(sample)


# 在更小的数据集上对结果进行测试
# "/home/cyx/Datasets/good_perfor_path.json"
class FN3D(Dataset):
    def __init__(self,root='', mode='train', data_json_path="/home/cyx/Datasets/Combied_Bra_UPENN_set.json",
                 labeled_json="/home/cyx/Datasets/labeled_data.json", 
                #  generate_data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/generate_texts_dic_data_two.json",  #
                 generate_data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/cleaned_with_info_final_fixed_text.json",  #
                #  generate_data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/generate_texts_dic_data_two_v7_final_text.json",  #
                 age_gender_info="/home/cyx/Datasets/All_with_age_and_gender2.json",
                 use_addition=False, use_bs=False,   # 0
                 text_only = False
                 ):

        use_wash = True
        # generate_data_path = "/home/cyx/Codes/M3D/best_texts_dic_ucsf.json"
        self.lines = []
        names, idhs = [], []
        self.use_addition = use_addition
        if self.use_addition:
            # if "UCSF" in data_json_path:
            # generate_data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/generate_texts_ucsf_v7.json"  #
            generate_data_path_ucsf = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/ucsf_cleaned_with_info_fixed_text.json"  #
            # generate_data_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/evaluated_texts/generate_texts_ucsf_dic_data_two.json"  #
            additional_texts = load_json(generate_data_path_ucsf)
            # if use_wash:
            #     generate_data_path = generate_data_path[:-5] + "_V1.json"

            self.texts = load_json(generate_data_path)
            self.texts = self.texts | additional_texts
            self.patient_info = load_json(age_gender_info)


        else:
            self.texts = None

        if "BraData" in root:
            self.use_brats = True
        else:
            self.use_brats = False

        # if mode == "valid":
        #     the_mode = "test"
        # elif mode == "test":
        #     the_mode = "valid"
        # else:
        the_mode = mode

        label = load_json(labeled_json)
        paths = load_json(data_json_path)[the_mode.capitalize()]



        for i in range(len(paths)):
            name = paths[i].split("/")[-1][:-4]
            idh = label[name]
            names.append(name)
            idhs.append(idh)

        # for i in range(len(paths)):
        #     paths[i] = paths[i][:-4] +"_resized"+paths[i][-4:]


        self.mode = mode
        self.names = names
        self.paths = paths
        self.idhs = idhs
        self.use_bs = use_bs
        self.text_only = text_only



    def __getitem__(self, item):
        # remove brain glioma
        # text_prompt_template = "Image description: \n {} \n Patient Status: \n The {} brain glioma patient is {} years old."
        text_prompt_template = "Patient Status: \n The {} brain glioma patient is {} years old. \n Image description: \n {}"
        # text_prompt_template = "The {} brain glioma patient is {} years old. \n\n\n {}"
        # text_prompt_template = "The brain glioma {} patient is {} years old."
        # text_prompt_t = "The brain glioma {} patient is {} years old. The {} modality image description is as follows: {}" modalities[ind] ,
        # text_prompt_template = "The {} modality image description is as follows: {}"
        # pt_template = "The brain glioma {} patient is {} years old."

        path = self.paths[item]
        name = self.names[item]
        idh = self.idhs[item]

        modalities = ["T1","T1ce","T2","Flair"]

        # if self.use_addition:
        #     age_gender = self.patient_info[name]
        #     text = self.texts[name]["texts"]
        #     # pt_prompt = "The brain glioma {} patient is {} years old.".format(age_gender[0],int(age_gender[1]))
        #     for ind in range(len(text)):
        #         text[ind] = text_prompt_template.format(age_gender[0],int(age_gender[1]),text[ind])
        #     # text.append(pt_template.format(age_gender[0],int(age_gender[1]),))

        if self.use_addition:
            age_gender = self.patient_info[name]
            # pt_prompt = "The brain glioma {} patient is {} years old.".format(age_gender[0],int(age_gender[1]))
            # text = self.texts[name]["0"]["text"]
            text = self.texts[name]
            text = text_prompt_template.format(age_gender[0],int(age_gender[1]),text)
            # text = text_prompt_template.format(text,age_gender[0],int(age_gender[1]))
            # text = text_prompt_template.format(age_gender[0],int(age_gender[1]))
            # for ind in range(len(text)):
            #     text[ind] = "The {} modality image description is as follows: {}".format(modalities[ind],text[ind])
            # text_pt = torch.load("/home/cyx/Datasets/Text_prompt/"+name+".pt")
        
        if self.text_only and self.use_addition:
            return text, torch.tensor(idh)
    
        if self.use_brats:
            data = pkload(path + 'data_f32b0.pkl')[0]
        else:
            data = np.load(path)

        if self.mode == 'train':
            # T1 T1ce T2 Flair
            # image = data[:,:,:,1][:,:,:,None]
            image = data
            sample = {'image': image}
            sample = transform(sample)
            # idh_target = grade[1]

            images = sample['image']
            label = torch.tensor(idh)

            # return sample['image'], torch.tensor(idh)

        elif self.mode == 'valid':

            # image = data[:,:,:,1][:,:,:,None]
            image = data
            sample = {'image': image}
            sample = transform_valid(sample)

            images = sample['image']
            label = torch.tensor(idh)
        else:
            # image = data[:,:,:,1][:,:,:,None]
            image = data
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')

            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()

            images = image
            label = torch.tensor(idh)

        if self.use_addition:
            return images,label,text

        return images,label


    def __len__(self):
        return len(self.names)

    # def collate(self, batch):
    #     return [torch.cat(v) for v in zip(*batch)]


if __name__ == "__main__":
    # train_root = "/home/cyx/Datasets/BraData/Train"
    # train_list = "/home/cyx/Datasets/BraData/Train/IDH_train_1.txt"

    train_set = FN3D(mode="valid", use_addition=True,
                     data_json_path="/home/cyx/Codes/data_process/Data_json/data_dic.json",use_bs=False)
    
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=4)
    #
    for i, data in enumerate(train_loader):
        print(data[0].shape)
        print(data[1])
        texts = data[2]
        print(texts)
        break
        # text_pt = data[3]
        # print(text_pt.shape)
        # print(texts[0])
        # need_texts = texts[0]
        # for text in texts[1:]:
        #     need_texts += text
        # for i in range(len(need_texts)):
        #     print(i,need_texts[i])
        # break