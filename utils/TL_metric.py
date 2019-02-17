import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class MetricNet(nn.Module):
    def __init__(self, feature_dim=256, num_class=0):
        super(MetricNet, self).__init__()
        self.num_class = num_class

        layer_dim = 128

        self.fc1 = nn.Linear(feature_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, layer_dim)
        self.out_layer = nn.Linear(layer_dim, self.num_class)
        init.normal_(self.out_layer.weight, std=0.001)
        init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.out_layer(out)
        return out


def metric_dist(track_feat, det_feat, metric_net):
    track_feat = np.array(track_feat)
    track_feat, det_feat = torch.from_numpy(track_feat).cuda().float(), torch.from_numpy(det_feat).cuda().float()
    m, n, dim = track_feat.size(0), det_feat.size(0), det_feat.size(1)
    track_feat = track_feat.view(-1, 1, dim).repeat(1, n, 1)
    det_feat = det_feat.view(1, -1, dim).repeat(m, 1, 1)
    dist = (track_feat - det_feat).abs().view(-1, dim)
    output = metric_net(dist)
    output = F.softmax(output, dim=1)
    score = output[:, 1] - output[:, 0]
    score = score.view(m, n).detach().cpu().numpy()
    return 1 - score
