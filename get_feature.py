import os
import h5py
import argparse

from datasets.mot_seq import get_loader
from utils.log import logger
from utils.feature_extractor import *


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('emb', data=results, dtype=float, maxshape=(None, None))
        pass


def get_seq_feat(dataloader, result_filename, save_dir=None, args=None):
    if save_dir is not None:
        mkdirs(save_dir)
    logger.info('Extracting {} feat for seq {}'.format('GT' if args.type == 'gt' else 'DET',
                                                       os.path.basename(result_filename)))
    seq_feat = FeatureExtractor(int(os.path.basename(result_filename).split('.')[0][-2:]), ide=args.ide)
    for frame_id, batch in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {}'.format(frame_id))
        frame, det_tlwhs, det_scores, gt_tlwhs, gt_ids = batch
        if args.type == 'gt':
            seq_feat.update(frame, gt_tlwhs, gt_ids)
        else:
            det_ids = -np.ones_like(det_scores)
            seq_feat.update(frame, det_tlwhs, det_ids)

    return seq_feat.lines


def main(data_root=os.path.expanduser('~/Data/MOT16/train'), det_root=None, seqs=('MOT16-05',), args=None):
    result_root = os.path.join(data_root, '..', '{}_feat'.format('gt' if args.type == 'gt' else 'det'),
                               '{}'.format('ide256' if args.ide else 'og512'))
    mkdirs(result_root)

    all_res = np.array([]).reshape(0, (512 if not args.ide else 256) + 3)

    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq)
        loader = get_loader(data_root, det_root, seq, num_workers=0)
        result_filename = os.path.join(result_root, 'features{}.h5'.format(seq))
        res = get_seq_feat(loader, result_filename, save_dir=output_dir, args=args)
        all_res = np.concatenate([all_res, res], axis=0)

    # save results
    write_results(os.path.join(result_root, 'all_seq_feat.h5'), all_res)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MOT Tracking")
    parser.add_argument('--ide', action='store_true')
    parser.add_argument('--type', type=str, choices=['det', 'gt'], default='gt')
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

    main(data_root=os.path.expanduser('~/Data/MOT16/train'), seqs=seqs, args=parser.parse_args())
