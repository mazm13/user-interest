import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm

import opts
from model import Ctr
from dataset import UserTest

if __name__ == "__main__":
    opt = opts.parse_opt()
    with open(os.path.join('models', 'le_user.param'), 'rb') as f:
        le_user = pickle.load(f)

    test_dataset = UserTest()
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=8)
    N = len(test_dataloader)

    model = Ctr(opt=opt).eval()
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join('models', 'model-3.ckpt')))

    luser_id, lphoto_id, lpred = [], [], []
    for i, data in enumerate(tqdm(test_dataloader)):
        user_id, visual_feature, photo_id = data
        user_id, visual_feature = user_id.cuda(), visual_feature.cuda()
        pred = model(user_id, visual_feature)
        pred = F.sigmoid(pred)

        luser_id.append(user_id)
        lphoto_id.append(photo_id)
        lpred.append(pred)

    print("Evaluation finished. Now save it into {} ...".format(os.path.join('results', 'submit.txt')))
    luser_id = torch.cat(luser_id, dim=0).detach().cpu().numpy()
    lphoto_id = torch.cat(lphoto_id, dim=0).detach().cpu().numpy()
    lpred = torch.cat(lpred, dim=0).squeeze(1).detach().cpu().numpy()

    luser_id = le_user.inverse_transform(luser_id)

    with open(os.path.join('results', 'submit.txt'), 'w') as f:
        for user_id, photo_id, pred in zip(tqdm(luser_id), lphoto_id, lpred):
            f.write('\t'.join([str(user_id), str(photo_id), '%.6f' % pred]) + '\n')
