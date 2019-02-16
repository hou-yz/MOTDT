import torch
import numpy as np
import torch.nn.functional as F


def metric_dist(track_feat, det_feat, metric_net):
    track_feat = np.array(track_feat)
    track_feat, det_feat = torch.from_numpy(track_feat).cuda().float(), torch.from_numpy(det_feat).cuda().float()
    m, n = track_feat.size(0), det_feat.size(0)
    track_feat = track_feat.view(-1, 1, 512).repeat(1, n, 1)
    det_feat = det_feat.view(1, -1, 512).repeat(m, 1, 1)
    dist = (track_feat - det_feat).abs().view(-1, 512)
    output = metric_net(dist)
    output = F.softmax(output, dim=1)
    score = output[:, 1] - output[:, 0]
    score = score.view(m, n).detach().cpu().numpy()
    return np.zeros(-score.shape)
