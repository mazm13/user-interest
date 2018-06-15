import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import pickle
import os

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

    with open(os.path.join('models', 'le_user.param'), 'wb') as f:
        pickle.dump(le_user, f)

    train_interaction['user_id'] = le_user.transform(train_interaction['user_id'])
    test_interaction['user_id'] = le_user.transform(test_interaction['user_id'])

    le_clic = LabelEncoder()
    le_clic.fit(train_interaction['click'])
    train_interaction['click'] = le_clic.transform(train_interaction['click'])

    train_interaction.to_csv('data/preprocessed/train_interaction.csv')
    test_interaction.to_csv('data/preprocessed/test_interaction.csv')

    print('Original Data has been initialized, {} for training and {} for testing) .'.format(train_size, test_size))
    return num_users


def face_reader(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            photo_id, json_serials = line.strip().split('\t')
            photo_id = int(photo_id)
            if photo_id in data.keys():
                assert ValueError("Photo_id {} appears again in {} .".format(photo_id, file_path))
            else:
                if json_serials != '0':
                    faces = json.loads(json_serials)
                    data[photo_id] = faces
    return data


def initialize_face_data():
    train_data = face_reader('./data/train/train_face.txt')
    test_data = face_reader('./data/test/test_face.txt')

    data = train_data.copy()
    data.update(test_data)

    array = []
    for key, values in data.items():
        # max scale index
        idx = 0
        for i, value in enumerate(values):
            if value[0] > values[idx][0]:
                idx = i
        data[key] = values[idx]
        array.append(values[idx])

    assert len(data.items()) == len(array)
    array = np.array(array)[:, [0, 2]]
    print("Data array shape: {}".format(array.shape))

    mean = array.mean(axis=0)
    std = array.std(axis=0)
    print("Mean and Std:{}, {}".format(mean, std))

    with open('./data/preprocessed/train_face.txt', 'a') as f:
        for key, _ in train_data.items():
            value = data[key]
            value[0] = (value[0] - mean[0]) / std[0]
            value[2] = (value[2] - mean[1]) / std[1]
            f.write(str(key) + '\t' + value.__str__() + '\n')

    with open('./data/preprocessed/test_face.txt', 'a') as f:
        for key, _ in test_data.items():
            value = data[key]
            value[0] = (value[0] - mean[0]) / std[0]
            value[2] = (value[2] - mean[1]) / std[1]
            f.write(str(key) + '\t' + value.__str__() + '\n')
    print("Face data preprocessed .")


if __name__ == "__main__":
    initialize_data()
