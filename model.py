import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceEmbedder(nn.Module):
    def __init__(self, opt):
        super(FaceEmbedder, self).__init__()
        self.opt = opt
        self.gender_embedder = nn.Embedding(2, opt.embed_dim)
        self.perp_embedder = nn.Embedding(opt.num_perps, opt.embed_dim)

    def forward(self, scale, gender, age, perp):
        gender_embedding = self.gender_embedder(gender).squeeze(1)
        perp_embedding = self.perp_embedder(perp).squeeze(1)
        ret = torch.cat([scale, gender_embedding, age, perp_embedding], dim=1)
        return ret


class IfClick(nn.Module):
    def __init__(self, opt):
        super(IfClick, self).__init__()
        self.opt = opt
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.user_dim)
        self.face_embedder = FaceEmbedder(opt)

        self.wh = nn.Linear(2 * opt.user_dim, opt.hidden_dim)
        self.wp = nn.Linear(opt.hidden_dim, 1)

    def forward(self, user_id, visual, scale, gender, age, perp):
        face_embedding = self.face_embedder(scale, gender, age, perp)
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
