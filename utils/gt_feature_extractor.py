import numpy as np
from models.reid import load_reid_model, extract_reid_features


def tlwh2tlbr(tlwh):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = tlwh.copy()
    ret[2:] += ret[:2]
    return ret


class FeatureExtractor(object):

    def __init__(self, cam):
        self.reid_model = load_reid_model()
        self.frame_id = 0
        self.cam = cam
        self.lines = np.array([]).reshape(0, 512 + 3)

    def update(self, image, tlwhs, pids):
        self.frame_id += 1

        # set features
        tlbrs = [tlwh2tlbr(tlwh) for tlwh in tlwhs]
        to_remove=[]

        for i in range(len(tlbrs)):
            tlbr = tlbrs[i]
            if (tlbr[1]<0 and tlbr[3]<0) or (tlbr[0]<0 and tlbr[2]<0):
                to_remove.append(i)
        pids = [pids[i] for i in range(len(tlbrs)) if i not in to_remove]
        tlbrs = [tlbrs[i] for i in range(len(tlbrs)) if i not in to_remove]

        features = extract_reid_features(self.reid_model, image, tlbrs)
        features = features.cpu().numpy()
        pids = np.array(pids, ndmin=2).transpose()
        line = np.concatenate([np.ones_like(pids) * self.cam, pids, np.ones_like(pids) * self.frame_id, features],
                              axis=1)
        self.lines = np.concatenate([self.lines, line], axis=0)
