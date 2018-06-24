import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

import resource
import os
import time

from model import IfClick, Ctr, DeepFM, DeepCross
from dataset import UserInter
import opts

from misc.logger import Logger
from misc.loss import VBCELoss, FocalLoss
from misc import utils

if __name__ == "__main__":
    opt = opts.parse_opt()
    # Fix: "RuntimeError: received 0 items of ancdata" when validating
    # From: "https://github.com/fastai/fastai/issues/23"
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # load dataset, and split it into training dataset and validating dataset
    dataset = UserInter()
    N = len(dataset)
    indices = list(range(N))
    split = int(np.floor(opt.valid_size * N))

    if opt.valid_shuffle:
        # np.random.seed(opt.random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=train_sampler,
        num_workers=16, pin_memory=False,
    )
    valid_loader = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=valid_sampler,
        num_workers=16, pin_memory=False,
    )

    M = len(train_loader)
    print(M)
    num_valid = len(valid_loader)
    print(num_valid)

    model = DeepCross(opt=opt)
    model = model.cuda()

    current_lr = 4e-4
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss()
    logger = Logger('./logs/')

    for epoch in range(opt.num_epoches):
        # schedule learning rate
        frac = epoch // 3
        decay_factor = 0.8 ** frac
        current_lr = current_lr * decay_factor
        utils.set_lr(optimizer, current_lr)

        # training
        model.train()
        start = time.time()

        for i, data in enumerate(train_loader):
            # prepare data and corresponding label(which is 'click')
            user_id = data['user_id'].cuda()
            hour = data['hour'].cuda()
            visual = data['visual'].cuda()
            click = data['click'].cuda()

            scale = data['scale'].cuda()
            gender = data['gender'].cuda().squeeze(1)
            age = data['age'].cuda().squeeze(1)
            attribute = data['attribute'].cuda().squeeze(1)

            # compute loss and backward gradient
            pred = model(user_id=user_id, hour=hour, visual=visual, scale=scale, gender=gender, age=age,
                         attribute=attribute)
            loss = criterion(pred, click)

            # backward
            optimizer.zero_grad()
            loss.backward()
            utils.clip_gradient(optimizer, 0.1)
            optimizer.step()

            if i % 50 == 0:
                end = time.time()
                auc_score = metrics.roc_auc_score(click.detach().cpu().numpy(), pred.detach().cpu().numpy()[:, 1])
                print("iter {}/{} (epoch {}), train_loss = {:.6f}, auc_score = {:.6f}, time/log = {:.3f}"
                      .format(i, M, epoch, loss.item(), auc_score, end - start))
                logger.scalar_summary('loss', loss.item(), i + epoch * M)
                logger.scalar_summary('train_auc_score', auc_score, i + epoch * M)
                start = time.time()

        # save model
        torch.save(model.state_dict(), os.path.join(opt.model_path, 'model-{}.ckpt'.format(epoch)))

        # Validating
        model.eval()
        y_, pred_ = [], []
        for i, data in enumerate(valid_loader):
            # prepare data and corresponding label(which is 'click')
            user_id = data['user_id'].cuda()
            hour = data['hour'].cuda()
            visual = data['visual'].cuda()
            click = data['click'].numpy()

            scale = data['scale'].cuda()
            gender = data['gender'].cuda().squeeze(1)
            age = data['age'].cuda().squeeze(1)
            attribute = data['attribute'].cuda().squeeze(1)

            pred = model(user_id=user_id, hour=hour, visual=visual, scale=scale, gender=gender, age=age,
                         attribute=attribute)
            pred = pred[:, 1]
            pred_.append(pred.detach().cpu().numpy())
            y_.append(click)
        y_ = np.concatenate(y_, axis=0).squeeze(1)
        pred_ = np.concatenate(pred_, axis=0)
        auc_score = metrics.roc_auc_score(y_, pred_)
        logger.scalar_summary('auc', auc_score, epoch)
