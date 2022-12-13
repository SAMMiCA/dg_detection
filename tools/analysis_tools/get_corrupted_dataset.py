# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.analysis_tools.robustness_eval import get_results

from mmdet import datasets
from mmdet.apis import multi_gpu_test, set_random_seed, save_data
from mmdet.core import eval_map
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


''''
How 2 run?
    you can set the parameters same as test_robustness.py !
    Note:   This code is simply written based on test_robustness.py
            Therefore, it was not completely focused on saving dataset-c.
            Therefore, unnecessary parameters or codes are included.
            Please set the required parameters by referring to save_data() function.
            You can just enter any value for the remaining parameters.
'''

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--severities',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5],
        help='corruption severity levels')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for pascal voc evaluation')
    parser.add_argument(
        '--summaries',
        type=bool,
        default=False,
        help='Print summaries for every corruption and severity')
    parser.add_argument(
        '--workers', type=int, default=32, help='workers per gpu')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--final-prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default='mPC',
        help='corruption benchmark metric to print at the end')
    parser.add_argument(
        '--final-prints-aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='benchmark',
        help='aggregate all results or only those for benchmark corruptions')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--load-dataset',
        type=str,
        choices=['original', 'corrupted'],
        default='original',
        help='Add Corrupt'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.show_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out", "--show" or "show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.workers == 0:
        args.workers = cfg.data.workers_per_gpu

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed)

    if 'all' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
            'saturate'
        ]
    elif 'benchmark' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
    elif 'noise' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in args.corruptions:
        corruptions = [
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        ]
    elif 'weather' in args.corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in args.corruptions:
        corruptions = [
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    elif 'holdout' in args.corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in args.corruptions:
        corruptions = ['None']
        args.severities = [0]
    else:
        corruptions = args.corruptions

    rank, _ = get_dist_info()
    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for sev_i, corruption_severity in enumerate(args.severities):
            # evaluate severity 0 (= no corruption) only once
            if corr_i > 0 and corruption_severity == 0:
                continue
            test_data_cfg = copy.deepcopy(cfg.data.test)
            # assign corruption and severity

            if args.load_dataset == 'original':
                if corruption_severity > 0:
                    corruption_trans = dict(
                        type='Corrupt',
                        corruption=corruption,
                        severity=corruption_severity)
                    # TODO: hard coded "1", we assume that the first step is
                    # loading images, which needs to be fixed in the future
                    test_data_cfg['pipeline'].insert(1, corruption_trans)
            else:
                raise NotImplementedError(
                    "The types of load_dataset can be 'original' or 'corrupted'.")

            # print info
            print(f'\nTesting {corruption} at severity {corruption_severity}')

            # build the dataloader
            # TODO: support multiple images per gpu
            #       (only minor changes are needed)
            dataset = build_dataset(test_data_cfg)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=args.workers,
                dist=distributed,
                shuffle=False)

            if not distributed:
                show_dir = args.show_dir
                if show_dir is not None:
                    show_dir = osp.join(show_dir, corruption)
                    show_dir = osp.join(show_dir, str(corruption_severity))
                    if not osp.exists(show_dir):
                        os.makedirs(show_dir)
                outputs = save_data(data_loader, win_name='', out_dir=show_dir,
                                    show_score_thr=args.show_score_thr)
            else:
                raise TypeError("It does not support distribution mode")


if __name__ == '__main__':
    main()
