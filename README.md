<div align="center">
<h1> LSDF-UNet: Lightweight Large-Small Network with Dual-Size Patch Frequency Aware for Medical Image Segmentation </h1>
</div>

##  News

- [2025.8.1] Training and inference code released

##  Abstract

Efficient and effective perception and aggregation mechanisms are crucial in medical image segmentation, especially in scenarios with limited computing resources. However, existing methods are usually accompanied by high computational costs. At the same time, the ubiquitous co-occurrence phenomenon makes it difficult for the model to effectively distinguish target features from interfering background information. To address the above problems, this paper proposes an LSDF-UNet segmentation model. First, a "looking at the big-focusing on the small" dynamic fusion mechanism is adopted to achieve lightweight and efficient feature enhancement under linear complexity. Secondly, a dual-size patch frequency aware module (DPFA) is designed, combined with a frequency-aware block (FAB) and a dual-scale patch partitioning strategy to separate high-frequency details and low-frequency contours in the frequency domain, suppress co-occurrence noise, and significantly improve boundary discrimination. Through extensive experiments on three benchmark medical image datasets, it is demonstrated that our method achieves state-of-the-art performance and effectiveness.

##  Introduction

<div align="center">
    <img width="800" alt="image" src="asserts/challen_.png?raw=true">
</div>

Major challenges in medical image segmentation.

##  Overview

<div align="center">
<img width="800" alt="image" src="asserts/LSDF-UNet.png?raw=true">
</div>

The overall architecture of LSDF-UNet.

##  TODO

- [x] Release code

##  Getting Started

### 1. Install Environment

```
conda create -n LSDF-UNet python=3.10
conda activate LSDF-UNet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), and Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the LSDF-UNet

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the LSDF-UNet

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```


##  Quantitative comparison

<div align="center">
<img width="800" alt="image" src="asserts/compara.jpg?raw=true">
</div>

<div align="center">
    Comparison with other methods on the ISIC2018, Kvasir and Moun-Seg datasets.
</div>



##  License

The content of this project itself is licensed under [LICENSE](https://github.com/Anonymous-Submission2025/NetWork/LSDF-UNet/blob/main/LICENSE).
