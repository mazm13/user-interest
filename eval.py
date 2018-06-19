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


def results_to_file(luser_id, lphoto_id, lpred, le_user):
    luser_id = torch.cat(luser_id, dim=0).cpu().numpy()
    lphoto_id = torch.cat(lphoto_id, dim=0).cpu().numpy()
    lpred = torch.cat(lpred, dim=0).cpu().numpy()

    luser_id = le_user.inverse_transform(luser_id)
    with open(os.path.join('results', 'submit.txt'), 'a') as f:
        for user_id, photo_id, pred in zip(luser_id, lphoto_id, lpred):
            f.write('\t'.join([str(user_id), str(photo_id), '%.6f' % pred]) + '\n')


if __name__ == "__main__":
    opt = opts.parse_opt()
    with open(os.path.join('models', 'le_user.param'), 'rb') as f:
        le_user = pickle.load(f)

    test_dataset = UserTest()
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=16)
    N = len(test_dataloader)

    model = Ctr(opt=opt).eval()
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join('models', 'model-5.ckpt')))

    # clean the submit file
    with open(os.path.join('results', 'submit.txt'), 'w') as f:
        f.write('')

    luser_id, lphoto_id, lpred = [], [], []
    for i, data in enumerate(tqdm(test_dataloader)):
        user_id, visual_feature, photo_id = data
        user_id, visual_feature = user_id.cuda(), visual_feature.cuda()
        pred = model(user_id, visual_feature)
        # pred = F.sigmoid
        pred = pred[:, 1]

        luser_id.append(user_id.detach())
        lphoto_id.append(photo_id.detach())
        lpred.append(pred.detach())

        # In case of out of memory error
        if (i + 1) % 500 == 0:
            results_to_file(luser_id, lphoto_id, lpred, le_user)
            luser_id, lphoto_id, lpred = [], [], []

    results_to_file(luser_id, lphoto_id, lpred, le_user)
