import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from model import IfClick
from preprocess import UserInter
import opts

from misc.logger import Logger

if __name__ == "__main__":
    train_data = UserInter()
    train_data = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)
    N = len(train_data)

    opt = opts.parse_opt()
    ifClick = IfClick(opt=opt)
    ifClick = ifClick.cuda()

    optimizer = optim.Adam(ifClick.parameters())
    criterion = nn.BCELoss()
    ifClick.train()
    logger = Logger('./logs/')

    for epoch in range(5):
        for i, data in enumerate(train_data):
            # prepare data and corresponding label(which is 'click')
            user_id = data[0].cuda()
            visual_feature = data[1].cuda()
            click = data[2].cuda().float()

            # compute loss and backward gradient
            pred = ifClick(user_id, visual_feature)
            loss = criterion(pred, click)
            ifClick.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Loss: {:.6f}'.format(loss.item()))
                logger.scalar_summary('loss', loss.item(), i + epoch * N)
