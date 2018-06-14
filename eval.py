import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

import opts
from model import Ctr

if __name__ == "__main__":
    opt = opts.parse_opt()
    with open(os.path.join('models', 'le_user.param'), 'rb') as f:
        le_user = pickle.load(f)

    test_interaction = pd.read_csv('./data/preprocessed/test_interaction.csv')
    test_interaction['user_oid'] = le_user.inverse_transform(test_interaction['user_id'])
    N = len(test_interaction)

    model = Ctr(opt=opt).eval()
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join('models', 'model-1.ckpt')))

    with open(os.path.join('results', 'submit.txt'), 'w') as f:
        for i in range(N):
            data = test_interaction.loc[i]
            user_id = data['user_id']
            user_id = torch.from_numpy(np.array(user_id)).cuda()
            photo_id = data['photo_id']
            photo_path = 'data/train/preliminary_visual_train/' + str(photo_id)
            visual_feature = torch.from_numpy(np.load(photo_path).squeeze()).cuda()

            pred = model(user_id, visual_feature)
            pred = F.sigmoid(pred)
            pred = pred.detach().cpu().numpy()

            user_id_ = data['user_oid']
            f.write('\t'.join([str(user_id_), str(photo_id), '%.6f' % pred]) + '\n')
