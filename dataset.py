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
        hour = torch.LongTensor(np.array(data['hour']))
        click = torch.LongTensor(np.array([data['click']]))

        # visual
        photo_id = data['photo_id']
        photo_path = '../dataset/preliminary_visual_train/' + str(photo_id)
        visual_feature = torch.from_numpy(np.load(photo_path).squeeze())

        # face
        if photo_id in self.face_data.keys():
            face_raw = self.face_data[photo_id]
        else:
            face_raw = [0, 0, 0, 20]
        scale = torch.FloatTensor([face_raw[0]])
        gender = torch.LongTensor([face_raw[1]])
        age = torch.LongTensor([face_raw[2]])
        attribute = torch.LongTensor([face_raw[3] - 20])

        return {'user_id': user_id, 'visual': visual_feature, 'hour': hour, 'click': click, 'scale': scale,
                'gender': gender, 'age': age, 'attribute': attribute}

    def __len__(self):
        return len(self.data)


class UserTest(Dataset):
    def __init__(self):
        data = pd.read_csv('./data/preprocessed/test_interaction.csv')
        self.data = data

    def __getitem__(self, item):
        data = self.data.loc[item]
        user_id = torch.from_numpy(np.array(data['user_id']))
        photo_id = data['photo_id']
        photo_path = '../dataset/preliminary_visual_test/' + str(photo_id)
        visual_feature = torch.from_numpy(np.load(photo_path).squeeze())
        return user_id, visual_feature, photo_id

    def __len__(self):
        return len(self.data)
