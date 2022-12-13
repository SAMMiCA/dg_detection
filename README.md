# DGDet: Domain Generalized Detection

## Introduction

DGDet is a method for improving performance in an out-of-distribution (OOD) environment while maintaining detection performance in an in-distribution environment.

This code is based on MMdetection open source.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Dependencies

Please refer to [get_started.md](docs/en/get_started.md) in MMdetection for installation.

## Getting Started

Please refer to [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection.

### Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [WandB](https://wandb.ai/)
- Pillow 7.2.0

We also provide docker files for building the environment.

- `docker pull dshong/mmdetection:1.6.0-cuda10.1-cudnn7`
- `docker pull dshong/mmdetection:1.8.0-cuda10.1-cudnn7`
- `docker pull dshong/mmdetection:1.10.2-cuda10.1-cudnn7`
- `docker pull dshong/mmdetection:1.11.0-cuda11.3`

**NOTE**: The docker file above does not contain WandB. 
To use WandB, please complete the installation through `pip install wandb` and login through `wandb login`.

### Running

The running settings are included in the configuration file. 
The configuration files are located in the `configs` folder.
You can train your model by passing keyword arguments to `train.py` as show below.

* **base** (8epochs)
  ```shell
  python3 /ws/external/tools/train.py /ws/external/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py --work-dir /ws/data/ai28/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes
  ```

* **w/ AugMix** (2epochs)
  ```shell
  python3 /ws/external/tools/train.py /ws/external/configs/cityscapes/2epoch/faster_rcnn_r50_fpn_1x_cityscapes_augmix.py --work-dir /ws/data/ai28/cityscapes/2epoch/faster_rcnn_r50_fpn_1x_cityscapes_augmix
  ```

* **DGDet (w/ AugMix)** (2epochs)
  ```shell
  python3 /ws/external/tools/train.py /ws/external/configs/cityscapes/2epoch/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100.py --work-dir /ws/data/ai28/cityscapes/2epoch/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100
  ```

The model can be tested in the same way as below.

* **base**
  ```shell
  python3 /ws/external/tools/test.py /ws/data/ai28/faster_rcnn_r50_fpn_1x_cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py /ws/data/ai28/faster_rcnn_r50_fpn_1x_cityscapes/epoch_8.pth  --work-dir /ws/external/ai28/faster_rcnn_r50_fpn_1x_cityscapes --eval bbox
  ```
  
* **DGDet (w/ AugMix)** (2epochs)
  ```shell
  python3 /ws/external/tools/test.py /ws/data/ai28/2epoch/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100.py /ws/data/ai28/2epoch/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100/epoch_2.pth  --work-dir /ws/external/ai28/2epoch/augmix.wotrans_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.100 --eval bbox
  ```

**NOTE:** Before running, make sure you have chosen the correct configuration file.

## Comparison

Results on CityScapes with Faster-RCNN:

|           | Detector    | Arch     | RPN Loss                           | RoI Loss                                 | Clean mAP(%) | Corruption mPC(%) |
|-----------|-------------|----------|------------------------------------|------------------------------------------|--------------|-------------------|
| Base      | Faster-RCNN | ResNet50 | CrossEntropy + L1loss              | CrossEntropy + SmoothL1loss              | 40.6         | 11.0              |
| w/ AugMix | Faster-RCNN | ResNet50 | CrossEntropy + L1loss              | CrossEntropy + SmoothL1loss              | 42.8         | 16.0              |
| DGDet     | Faster-RCNN | ResNet50 | CrossEntropyPlus(jsdv1.3) + L1loss | CrossEntropyPlus(jsdv1.3) + SmoothL1loss | 40.2         | 20.9              |

## Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
