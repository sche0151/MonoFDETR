# MonoFDETR: Monocular 3D Object Detection with Fine‐grained Depth‐guided from Zero‐Shot Depth Priors

This repository hosts the official implementation of [MonoFDETR: Monocular 3D Object Detection with Fine‐grained Depth‐guided from Zero‐Shot Depth Priors] based on the excellent work of [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) and [MonoDGP](https://github.com/PuFanqi23/MonoDGP).

In this work, we propose a monocular 3D object detection method using a fine-grained depth-guided transformer, named MonoFDETR. We replace the depth predictor with a pre-trained zero-shot depth estimation model and acquire enhanced visual embeddings and depth embeddings enriched with depth cues through a vision-depth fusion module. Then, we eliminate explicit depth map supervision and employ implicit depth position embeddings to capture global depth relations. Moreover, MonoFDETR integrates geometric depth with its error as a more rational depth prediction approach, based on analyzing the distribution of the predicted depth data. To further improve the transformer fused input representation, we apply mixup3D to depth maps and emphasize object regions using a region segmentation technique. Our fine-grained depth-guided method, which requires no additional data, attains state-of-the-art results on the KITTI benchmark with monocular images.

![](https://s2.loli.net/2025/06/20/lxMjBwgRkINFEyh.jpg)

The official results in the paper:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoDGP</td>
        <td div align="center">32.66%</td> 
        <td div align="center">23.69%</td> 
        <td div align="center">20.24%</td> 
    </tr>  
</table>

New and better results in this repo:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>
        <td rowspan="2",div align="center">Ckpts</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoDGP</td>
        <td div align="center">32.66%</td> 
        <td div align="center">23.69%</td> 
        <td div align="center">20.24%</td> 
        <td div align="center"><a href="https://pan.quark.cn/s/3f303e2fedd3">ckpt</a></td>
    </tr>  
</table>

Test results submitted to the official KITTI Benchmark:

Car category:

## MonoFDETR Codebase

### 1. Enviroment and Installation

#### 1.1 Clone this project and create a conda environment: python>=3.8, our version of python == 3.12.8

#### 1.2 Install pytorch and torchvision matching your CUDA version: torch >= 1.9.0, our version of torch == 2.5.0, pytorch-cuda==11.8

#### 1.3 Install Huggingface accelerate, transformers, and diffusers: accelerate >= 1.2.1, transformers >= 4.46.3, our version of accelerate == 1.2.1, transformers == 4.46.3

```bash
conda install -c conda-forge accelerate transformers
```

#### 1.4 Install other dependencies

```bash
conda install numba
conda install -c conda-forge scikit-image matplotlib-base
pip install albumentations timm ninja wandb
```

#### 1.5 Download [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and prepare the directory structure as follows:

```bash
    |MONOFDETR
    |──...
    |data/KITTIDataset
    |──ImageSets/
    |──training/
    |   ├──image_2
    |   ├──label_2
    |   ├──calib
    |──testing/
    |   ├──image_2
    |   ├──calib
```

You can also change the data path at "dataset/root_dir" in `configs/monofdetr.yaml`.

### 1.6 Download pre-trained model and checkpoint

You should download the Pre-trained model of **MonoDETR** from [MonoDETR](https://drive.google.com/file/d/1d8fbAt-CQF-IN8UEHuw3NimmfONhH6iA/view?usp=sharing) , and place it in the `pretrained-models` directory. You can also change this path at "monodetr_model" in `configs/monofdetr.yaml`.

You should download the Pre-trained model of **Depth Anything V2** from [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) , and place it in the `pretrained-models` directory. You can also change this path at "model/pretrained_dpt_path" in `configs/monofdetr.yaml`.

The Pre-trained model of **backbone** will auto download by `timm` [PyTorch Image Models](https://huggingface.co/timm) , or you can change the backbone version at "pretrained_backbone_path" in `configs/monofdetr.yaml`.

### 2. Get Strated

#### 2.1 Train

You can modify the settings of GPU, models and training in `configs/monofdetr.yaml`
```vim
python train.py --config configs/monofdetr.yaml
```

#### 2.2 Test
The best checkpoint will be evaluated as default. You can change it at "pretrain_model: 'checkpoint_best.pth'" in configs/monofdetr.yaml:
```vim
python test.py --config configs/monofdetr.yaml
```

#### 2.3 DDP Train
Our model supports [Accelerate](https://huggingface.co/docs/accelerate/quicktour), which offers a unified interface for launching and training on different distributed setups. This allows you to easily scale your PyTorch code for training and inference on distributed setups with hardware like GPUs and TPUs.

Firstly, set DDP config with a unified interface:
```vim
accelerate config
```
You can check your setup:
```vim
accelerate test
```
When the DDP setup completed, enjoying the distributed train:
```vim
accelerate launch train.py --config configs/monofdetr.yaml
```

## Citation

```
```

## Acknowlegment

Our code is based on (ICCV 2023)[MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), We sincerely appreciate their contributions and authors for releasing source codes. 
