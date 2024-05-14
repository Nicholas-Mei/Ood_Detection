# Multi-Label Out-of-Distribution Detection with Spectral Normalized Joint Energy

This is a [PyTorch](http://pytorch.org) implementation of [Multi-Label Out-of-Distribution Detection with Spectral Normalized Joint Energy](https://arxiv.org/abs/2405.04759) by Yihan Mei, Xinyu Wang, Dell Zhang, Xiaoling Wang.
Code is modified from [JointEnergy](https://github.com/deeplearning-wisc/multi-label-ood), [ODIN](https://github.com/facebookresearch/odin),  [Outlier Exposure](https://github.com/hendrycks/outlier-exposure), and [deep Mahalanobis
detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector).

## Datasets

Our experimental configuration of in-distribution and out-of-distribution datasets are almost identical with [JointEnergy](https://github.com/deeplearning-wisc/multi-label-ood).

Put PASCAL-VOC under `./Pascal/` folder, and put Texture under `./dtd/` folder.

## Training the Model

Train the ResNet model for PASCAL-VOC dataset

`python train.py --arch resnet101 --dataset pascal --save_dir ./save_models/`

## OoD Detection

To reproduce the SNoJoE score for PASCAL-VOC dataset, please run: 

`python eval.py --arch resnet101 --dataset pascal --ood_data imagenet --ood energy --method sum`

## OoD Detection Result

OoD detection performance comparison using SNoJoE vs. competitive
baselines.

<img src="./pic/result.png" alt="result" />



## Citation

```
@misc{mei2024multilabel,
      title={Multi-Label Out-of-Distribution Detection with Spectral Normalized Joint Energy}, 
      author={Yihan Mei and Xinyu Wang and Dell Zhang and Xiaoling Wang},
      year={2024},
      eprint={2405.04759},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

