import torch
import torch.nn as nn
import torch.nn.functional as F


class IfClick(nn.Module):
    def __init__(self, opt):
        super(IfClick, self).__init__()
        self.opt = opt
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.user_dim)
        self.wh = nn.Linear(2 * opt.user_dim, opt.hidden_dim)
        self.wp = nn.Linear(opt.hidden_dim, 1)

    def forward(self, user_id, visual):
        lvec = self.user_embedder(user_id)
        rvec = self.visu_embedder(visual)

        mul_d = torch.mul(lvec, rvec)
        abs_d = torch.abs(torch.add(lvec, -rvec))
        vec_d = torch.cat((mul_d, abs_d), dim=1)

        out = F.sigmoid(self.wh(vec_d))
        out = F.sigmoid(self.wp(out))
        # out = F.softmax(self.wp(out), dim=1)
        # out = F.cosine_similarity(lvec, rvec, dim=2)
        return out
