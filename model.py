import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceEmbedder(nn.Module):
    def __init__(self, opt):
        super(FaceEmbedder, self).__init__()
        self.opt = opt
        self.gender_embedder = nn.Embedding(2, opt.face_k)
        self.attribute_embedder = nn.Embedding(opt.num_attributes, opt.face_k)
        self.age_embedder = nn.Embedding(opt.num_ages, opt.face_k)

    def forward(self, scale, gender, age, attribute):
        gender_embedding = self.gender_embedder(gender)
        attribute_embedding = self.attribute_embedder(attribute)
        age_embedding = self.age_embedder(age)
        face = torch.stack([gender_embedding, attribute_embedding, age_embedding], dim=1)
        face = scale.unsqueeze(dim=1) * face
        return face


class IfClick(nn.Module):
    def __init__(self, opt):
        super(IfClick, self).__init__()
        self.opt = opt
        self.hour_embedder = nn.Embedding(24, opt.embed_dim)
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.embed_dim)
        self.face_embedder = FaceEmbedder(opt)

        self.fm_linear = nn.Linear(6 * opt.embed_dim, 2)
        self.dnn_linear_1 = nn.Linear(6 * opt.embed_dim, opt.hidden_dim)
        self.dnn_linear_2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.dnn_linear_3 = nn.Linear(opt.hidden_dim, 2)

    def forward(self, user_id, hour, visual, scale, gender, age, attribute):
        user = self.user_embedder(user_id)  # batch_size * user_dim
        hour = self.hour_embedder(hour)  # batch_size * embed_dim
        face = self.face_embedder(scale, gender, age, attribute)  # batch_size * 3 * embed_dim
        visual = self.visu_embedder(visual)  # batch_size * embed_dim
        embeddings = torch.cat([user.unsqueeze(1), hour.unsqueeze(1), visual.unsqueeze(1), face],
                               dim=1)  # batch_size * 6 * embed_dim

        batch_size = user_id.size(0)
        # FM part
        linear_part = self.fm_linear(embeddings.view(batch_size, 6 * self.opt.embed_dim))
        second_order = torch.pow(embeddings.sum(dim=1), 2) - torch.pow(embeddings, 2).sum(dim=1)
        second_order = 0.5 * second_order.sum(dim=1)
        fm = linear_part + second_order.unsqueeze(1)

        # DNN part
        x = F.relu(self.dnn_linear_1(embeddings.view(batch_size, 6 * self.opt.embed_dim)))
        x = F.relu(self.dnn_linear_2(x))
        x = self.dnn_linear_3(x)

        out = F.softmax(fm + x)
        return out


class Ctr(nn.Module):
    def __init__(self, opt):
        super(Ctr, self).__init__()
        self.opt = opt
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.embed_dim)
        self.fc1 = nn.Linear(opt.user_dim + opt.embed_dim, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hidden_dim, 2)

    def forward(self, user_id, visual):
        user = self.user_embedder(user_id)
        visual = self.visu_embedder(visual)
        x = torch.cat([user, visual], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(x))
        return out
