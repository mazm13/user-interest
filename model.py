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
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.embed_dim)
        self.face_embedder = FaceEmbedder(opt)

        self.fc1 = nn.Linear(33, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hidden_dim, 1)

    def forward(self, user_id, visual, scale, gender, age, perp):
        face = self.face_embedder(scale, gender, age, perp)  # batch_size * (2+2*embed_dim)
        user = self.user_embedder(user_id)  # batch_size * user_dim
        visual = self.visu_embedder(visual)  # batch_size * embed_dim

        x = torch.cat([user, visual, face], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class Ctr(nn.Module):
    def __init__(self, opt):
        super(Ctr, self).__init__()
        self.opt = opt
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.embed_dim)
        self.fc1 = nn.Linear(opt.user_dim + opt.embed_dim, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hidden_dim, 1)

    def forward(self, user_id, visual):
        user = self.user_embedder(user_id)
        visual = self.visu_embedder(visual)
        x = torch.cat([user, visual], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
