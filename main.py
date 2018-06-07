from torch.utils.data import DataLoader
import preprocess
import pandas as pd

#columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
#train_interaction = pd.read_table('data/train/' + 'train_interaction.txt', header=None)
# train_interaction.columns = columns
train_interaction = pd.read_csv('./data/preprocessed/train_interaction.csv')
N = len(train_interaction)

vs = {}
click = train_interaction['click']
for i in range(N):
    v = click[i]
    if v in vs.keys():
        vs[v] += 1
    else:
        vs[v] = 0

print(vs)