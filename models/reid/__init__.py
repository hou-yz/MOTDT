import cv2
import os
import numpy as np
from distutils.version import LooseVersion
import torch
from torch.autograd import Variable

from utils import bbox as bbox_utils
from utils.log import logger
from models import net_utils
from models.reid.image_part_aligned import Model
from models.reid.IDE_model import IDE_model


def load_reid_model(ide=False):
    if not ide:
        model = Model(n_parts=8)
        model.inp_size = (80, 160)
        ckpt = os.path.expanduser('~/Data/googlenet_part8_all_xavier_ckpt_56.h5')
        net_utils.load_net(ckpt, model)
        logger.info('Load ReID model from {}'.format(ckpt))
        model = model.cuda()
        model.eval()
        return model
    else:
        model = IDE_model(dropout=0.5, last_stride=1)
        model.inp_size = (128, 384)
        path = '/home/houyz/Code/open-reid-hyz/logs/ide_new/256/duke_my_gt/train/1_fps/basis/model_best.pth.tar'
        checkpoint = torch.load(path)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if True:
            keys_to_del = []
            for key in pretrained_dict.keys():
                if 'fc' in key:
                    keys_to_del.append(key)
            for key in keys_to_del:
                del pretrained_dict[key]
            pass
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        logger.info('Load ReID model from {}'.format(path))
        model = model.cuda()
        model.eval()
        return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def extract_reid_features(reid_model, image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)
    patches = np.asarray([im_preprocess(cv2.resize(p, reid_model.inp_size)) for p in patches], dtype=np.float32)

    gpu = net_utils.get_device(reid_model)
    if LooseVersion(torch.__version__) > LooseVersion('0.3.1'):
        with torch.no_grad():
            im_var = Variable(torch.from_numpy(patches))
            if gpu is not None:
                im_var = im_var.cuda(gpu)
            features = reid_model(im_var).data
    else:
        im_var = Variable(torch.from_numpy(patches), volatile=True)
        if gpu is not None:
            im_var = im_var.cuda(gpu)
        features = reid_model(im_var).data

    return features
