import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from misc.loss import one_hot
import math


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
        face = scale.unsqueeze(dim=1) * F.tanh(face)
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
        print(embeddings.size())
        exit(0)

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


class DeepFM(nn.Module):
    def __init__(self, opt):
        super(DeepFM, self).__init__()
        self.opt = opt
        self.hour_embedder = nn.Embedding(24, opt.face_k)
        self.user_embedder = nn.Embedding(opt.num_users, opt.face_k)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.face_k)
        self.face_embedder = FaceEmbedder(opt)

        self.fm_linear = nn.Linear(opt.num_users + 24 + 2 + opt.num_ages + opt.num_attributes, 2)
        self.dnn_linear_1 = nn.Linear(6 * opt.face_k, opt.hidden_dim)
        self.dnn_linear_2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.dnn_linear_3 = nn.Linear(opt.hidden_dim, 2)

    def forward(self, user_id, hour, visual, scale, gender, age, attribute):
        oh_user = one_hot(user_id, self.opt.num_users)
        oh_hour = one_hot(hour, 24)
        oh_gender = one_hot(gender, 2)
        oh_age = one_hot(age, self.opt.num_ages)
        oh_attribute = one_hot(attribute, self.opt.num_attributes)
        oh = torch.cat([oh_user, oh_hour, oh_gender, oh_age, oh_attribute], dim=1)

        # Linear part
        linear_part = self.fm_linear(oh)

        # FM part
        user = self.user_embedder(user_id)  # batch_size * user_dim
        hour = self.hour_embedder(hour)  # batch_size * embed_dim
        face = self.face_embedder(scale, gender, age, attribute)  # batch_size * 3 * embed_dim
        embeddings = torch.cat([user.unsqueeze(1), hour.unsqueeze(1), face], dim=1)  # batch_size * 6 * embed_dim

        second_order = torch.pow(embeddings.sum(dim=1), 2) - torch.pow(embeddings, 2).sum(dim=1)
        second_order = 0.5 * second_order.sum(dim=1)
        fm = linear_part + second_order.unsqueeze(1)

        # DNN part
        visual = self.visu_embedder(visual)  # batch_size * embed_dim
        batch_size = user_id.size(0)
        x = torch.cat([embeddings.view(batch_size, 5 * self.opt.face_k), visual], dim=1)
        x = F.relu(self.dnn_linear_1(x))
        x = F.relu(self.dnn_linear_2(x))
        x = self.dnn_linear_3(x)

        out = F.softmax(fm + x, dim=1)
        return out


class DeepCross(nn.Module):
    def __init__(self, opt):
        super(DeepCross, self).__init__()
        self.opt = opt
        self.hour_embedder = nn.Embedding(24, opt.hour_dim)
        self.user_embedder = nn.Embedding(opt.num_users, opt.user_dim)
        self.visu_embedder = nn.Linear(opt.visual_dim, opt.embed_dim)
        self.face_embedder = FaceEmbedder(opt)
        # Cross part
        x_len = opt.user_dim + opt.embed_dim + opt.hour_dim + opt.face_k * 3
        # x_len = 256 + 256 + 64 + 64 * 3
        self.red_1 = nn.Linear(x_len, 1, bias=False)
        self.bias_1 = Parameter(torch.Tensor(x_len))
        self.red_2 = nn.Linear(x_len, 1, bias=False)
        self.bias_2 = Parameter(torch.Tensor(x_len))
        self.red_3 = nn.Linear(x_len, 1, bias=False)
        self.bias_3 = Parameter(torch.Tensor(x_len))
        # Deep part
        self.fc_1 = nn.Linear(x_len, opt.hidden_dim)
        self.fc_2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc_3 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        # Classification part
        self.clf = nn.Linear(x_len + opt.hidden_dim, 2)
        # self.clf = nn.Linear(opt.hidden_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.bias_1.size(0))
        self.bias_1.data.uniform_(-stdv, stdv)
        self.bias_2.data.uniform_(-stdv, stdv)
        self.bias_3.data.uniform_(-stdv, stdv)

    def forward(self, user_id, hour, visual, scale, gender, age, attribute):
        batch_size = user_id.size(0)
        face = self.face_embedder(scale, gender, age, attribute)
        face = face.view(batch_size, self.opt.face_k * 3)
        hour = self.hour_embedder(hour)
        visual = self.visu_embedder(visual)
        user = self.user_embedder(user_id)

        # Cross part
        x_0 = torch.cat([user, visual, hour, face], dim=1)
        x_1 = x_0 * self.red_1(x_0) + self.bias_1 + x_0
        x_2 = x_0 * self.red_2(x_1) + self.bias_2 + x_1
        x_3 = x_0 * self.red_3(x_2) + self.bias_3 + x_2

        # Deep part
        h_1 = F.relu(self.fc_1(x_0))
        h_2 = self.fc_2(h_1)
        h_3 = F.relu(self.fc_3(h_2))

        out = torch.cat([x_3, h_3], dim=1)
        # print(out.size())
        out = self.clf(out)
        # print(out.size())
        out = F.softmax(out, dim=1)
        # print(out.size())
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
