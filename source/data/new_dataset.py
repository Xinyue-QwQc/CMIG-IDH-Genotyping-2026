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
        # modality num D H W
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
class FN3D(Dataset):
    def __init__(self,root='', mode='train', data_json_path="/home/cyx/Datasets/reset_3d_used.json",
                 labeled_json="/home/cyx/Datasets/labeled_data.json",
                 age_gender_info="/home/cyx/Datasets/All_with_age_and_gender2.json",
                 use_addition=False, language_template=0
                 ,csv_file='addition_data.csv', constraint_ratio=1):
        self.lines = []
        names, idhs = [], []
        self.use_addition = use_addition
        if self.use_addition:
            self.language_template = language_template
            patient_info = load_json(age_gender_info)
            age_genders = []

        if "BraData" in root:
            self.use_brats = True
        else:
            self.use_brats = False

        # if mode == "train":
        #     c_wild = int(463*constraint_ratio)
        #     c_Mutant = int(149*constraint_ratio)
        # elif mode == "valid":
        #     c_wild = int(99*constraint_ratio)
        #     c_Mutant = 32
        # else:
        #     c_wild = int(100*constraint_ratio)
        #     c_Mutant = 33

        label = load_json(labeled_json)
        paths = load_json(data_json_path)[mode.capitalize()]



        for i in range(len(paths)):
            name = paths[i].split("/")[-1][:-4]
            idh = label[name]
            names.append(name)
            idhs.append(idh)
            if self.use_addition:
                age_genders.append(patient_info[name])

        # for i in range(len(paths)):
        #     paths[i] = paths[i][:-4] +"_resized"+paths[i][-4:]


        self.mode = mode
        self.names = names
        self.paths = paths
        self.idhs = idhs

        if self.use_addition:
            self.age_genders = age_genders


    def __getitem__(self, item):
        path = self.paths[item]
        name = self.names[item]
        idh = self.idhs[item]

        langauge_template = ["The {} brain glioma patient is {} years old.",
                             "The {} patient is {} years old."
                              "{} is {} years old and suffers from brain glioma."]

        if self.use_addition:
            age_gender = self.age_genders[item]
            text = langauge_template[self.language_template].format(age_gender[0], age_gender[1])

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

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


if __name__ == "__main__":
    # train_root = "/home/cyx/Datasets/BraData/Train"
    # train_list = "/home/cyx/Datasets/BraData/Train/IDH_train_1.txt"

    train_set = FN3D(mode="test", use_addition=False)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=4)

    for i, data in enumerate(train_loader):
        print(data[0].shape)
        print(data[1])
        # print(data[2])
        if i == 3:
            break
    # print(len(train_set))
    # for i in range(1,17):
    #     train_loader = DataLoader(train_root,num_workers=i,batch_size=8,pin_memory=True,shuffle=True)
    #     a = time.time()
    #     for epoch in range(1,3):
    #         for k,data in enumerate(train_set):
    #             pass
    #     b = time.time()
    #     print(f"Spend time is {(b-a)/60} minute, num_wokers is {i}")
    # # pass
    # '/home/cyx/Datasets/BraData/'  'Train'  'IDH_valid_1.txt'
    # valid_list = '/home/cyx/Datasets/BraData/Train/IDH_valid_1.txt'
    # valid_root = '/home/cyx/Datasets/BraData/Train'
    # valid_set = HDT(valid_list, valid_root, 'valid')
    # valid_loader = DataLoader(valid_set, batch_size=1,)
    # for i,data in enumerate(valid_loader):
    #     print(i)


