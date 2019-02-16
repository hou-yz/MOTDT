import os
import h5py

from datasets.mot_seq import get_loader
from utils.log import logger
from utils.gt_feature_extractor import *


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('emb', data=results, dtype=float, maxshape=(None, None))
        pass


def eval_seq(dataloader, result_filename, save_dir=None):
    if save_dir is not None:
        mkdirs(save_dir)
    logger.info('Extracting GT feat for seq {})'.format(os.path.basename(result_filename)))
    gt_seq = FeatureExtractor(int(os.path.basename(result_filename).split('.')[0][-2:]))
    for frame_id, batch in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {})'.format(frame_id))
        frame, det_tlwhs, det_scores, gt_tlwhs, gt_ids = batch
        gt_seq.update(frame, gt_tlwhs, gt_ids)

    return gt_seq.lines



def main(data_root=os.path.expanduser('~/Data/MOT16/train'), det_root=None, seqs=('MOT16-05',)):
    result_root = os.path.join(data_root, '..', 'gt_feat')
    mkdirs(result_root)

    all_res = np.array([]).reshape(0, 512 + 3)

    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq)
        loader = get_loader(data_root, det_root, seq, num_workers=0)
        result_filename = os.path.join(result_root, 'features{}.h5'.format(seq))
        res = eval_seq(loader, result_filename, save_dir=output_dir)
        all_res = np.concatenate([all_res, res], axis=0)

    # save results
    write_results(os.path.join(result_root, 'all_seq_feat.h5'), all_res)
    pass

if __name__ == '__main__':
    # import fire
    # fire.Fire(main)

    seqs_str = '''
                MOT16-02
                MOT16-04
                MOT16-05
                MOT16-09
                MOT16-10
                MOT16-11
                MOT16-13'''
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=os.path.expanduser('~/Data/MOT16/train'), seqs=seqs)
