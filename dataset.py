import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from preprocess import face_reader

DATA_DIR = './data/'


class UserInter(Dataset):
    def __init__(self):
        data = pd.read_csv(DATA_DIR + 'preprocessed/train_interaction.csv')
        self.data = data
        face_data = face_reader(DATA_DIR + 'preprocessed/train_face.txt')
        self.face_data = face_data

    def __getitem__(self, item):
        data = self.data.loc[item]
        user_id = torch.from_numpy(np.array(data['user_id']))
        click = torch.from_numpy(np.array(data['click']))
        photo_id = data['photo_id']
        photo_path = DATA_DIR + 'train/preliminary_visual_train/' + str(photo_id)
        visual_feature = torch.from_numpy(np.load(photo_path).squeeze())

        if photo_id in self.face_data.keys():
            face_raw = self.face_data[photo_id]
        else:
            face_raw = [0, 0, 0, 0]
        scale = torch.Tensor([face_raw[0]])
        gender = torch.LongTensor([face_raw[1]])
        age = torch.Tensor([face_raw[2]])
        perp = torch.LongTensor([face_raw[3]])

        return {'user_id': user_id, 'visual': visual_feature, 'click': click, 'scale': scale, 'gender': gender,
                'age': age, 'prep': perp}

    def __len__(self):
        return len(self.data)
