import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = './data/'
RAW = ['user_id', 'click']


def initialize_data():
    columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    train_interaction = pd.read_table('data/train/' + 'train_interaction.txt', header=None)
    train_interaction.columns = columns
    train_size = len(train_interaction)

    test_columns = ['user_id', 'photo_id', 'time', 'duration_time']
    test_interaction = pd.read_table('data/test/' + 'test_interaction.txt', header=None)
    test_interaction.columns = test_columns
    test_size = len(test_interaction)

    le_user = LabelEncoder()
    le_user.fit(train_interaction['user_id'])
    num_users = len(le_user.classes_)
    print('UserId LabelEncoder Established, find {} users .'.format(num_users))

    train_interaction['user_id'] = le_user.transform(train_interaction['user_id'])
    test_interaction['user_id'] = le_user.transform(test_interaction['user_id'])

    le_clic = LabelEncoder()
    le_clic.fit(train_interaction['click'])
    train_interaction['click'] = le_clic.transform(train_interaction['click'])

    train_interaction.to_csv('data/preprocessed/train_interaction.csv')
    test_interaction.to_csv('data/preprocessed/test_interaction.csv')

    print('Original Data has been initialized, {} for training and {} for testing) .'.format(train_size, test_size))
    return num_users


class UserInter(Dataset):
    def __init__(self):
        data = pd.read_csv(DATA_DIR + 'preprocessed/train_interaction.csv')
        self.data = data

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


if __name__ == "__main__":
    initialize_data()
