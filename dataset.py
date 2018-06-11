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
        face_data = face_reader(DATA_DIR + 'train/train_face.txt')
        self.face_data = data

    def __getitem__(self, item):
        data = self.data.loc[item]
        user_id = torch.from_numpy(np.array(data['user_id']))
        click = torch.from_numpy(np.array(data['click']))
        photo_id = data['photo_id']
        photo_path = DATA_DIR + 'train/preliminary_visual_train/' + str(photo_id)
        visual_feature = torch.from_numpy(np.load(photo_path).squeeze())
        return [user_id, visual_feature, click]

    def __len__(self):
        return len(self.data)
