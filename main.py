from torch.utils.data import DataLoader
from dataset import UserInter
import preprocess
import pandas as pd
import json

columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table('data/train/' + 'train_interaction.txt', header=None)
train_interaction.columns = columns
# train_interaction = pd.read_csv('./data/preprocessed/train_interaction.csv')
# N = len(train_interaction)

#dataset = pd.read_csv('./data/preprocessed/train_interaction.csv')
face_data = preprocess.face_reader('./data/train/train_face.txt')

dataset = train_interaction
photo_ids = dataset['photo_id']
print(len(photo_ids))
photo_ids = photo_ids.unique()
print(len(photo_ids))

print(len(face_data))

not_in = []
for i in photo_ids:
    if i not in face_data.keys():
        not_in.append(i)

print(len(not_in))

