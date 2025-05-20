# Repository_of_MC4AD 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/examining-the-source-of-defects-from-a/3d-anomaly-detection-on-anomaly-shapenet)](https://paperswithcode.com/sota/3d-anomaly-detection-on-anomaly-shapenet?p=examining-the-source-of-defects-from-a)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/examining-the-source-of-defects-from-a/3d-anomaly-detection-on-real-3d-ad)](https://paperswithcode.com/sota/3d-anomaly-detection-on-real-3d-ad?p=examining-the-source-of-defects-from-a)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/examining-the-source-of-defects-from-a/3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/3d-anomaly-detection-and-segmentation-on?p=examining-the-source-of-defects-from-a)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/examining-the-source-of-defects-from-a/3d-anomaly-detection-on-anomaly-shapenet10)](https://paperswithcode.com/sota/3d-anomaly-detection-on-anomaly-shapenet10?p=examining-the-source-of-defects-from-a)

Hanzhe Liang, Aoran Wang, Jie Zhou, Xin Jin, Can Gao*, Jinbao Wang*  

\* Co-Corresponding

For the paper, you can check the following [link](https://arxiv.org/abs/2505.05901v2).

![image](https://github.com/hzzzzzhappy/MC4AD/blob/main/overview.png)

## Environments
### You need to create our environment. Our code runs on the device CUDA 11.1. If you do not have any environments already created, please run:
```bash
conda create -n py3-TA python=3.8
conda activate py3-TA
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision
conda install openblas-devel -c anaconda
```
Or you can run the following to create the environment:
```bash
conda env create -f environment.yml
```
Then, you need to download the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). Please run:
```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
## Datasets
Please download the datasets we used. You can download [Real3D-AD dataset](https://github.com/M-3LAB/Real3D-AD?tab=readme-ov-file) and [Anomaly-ShapeNet&Anomaly-ShapeNet-New](https://github.com/Chopper-233/Anomaly-ShapeNet). We provide dataload for Real3D-AD, Anomaly-ShapeNet, and Anomaly-ShapeNet-New. if you want to implement [MvTec3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), you need to go to the official implementation of [M3DM](https://github.com/nomewang/M3DM/blob/main/m3dm_runner.py), which is attributed to the fact that MvTec3D-AD uses depth maps, and you need to use the preprocessing tools and conversion tools it provide to change it to a point cloud form. Put the data and checkpoints in the corresponding folders:
```bash
code
├── datasets
│   ├── AnomalyShapeNet
│   │   ├── dataset
│   │   │   ├── obj
│   │   │   ├── pcd
├── log
│   ├── ashtray0
│   │   ├── ashtray0.pth
│   ├── ...
```
## Training & Evaluation
### Eval

You can train and change the train setting depending on ```config/train_config```:
```bash
python train.py --category bowl4 --logpath ./log/cap0/
# We take class cap0 for example; you can also replace 'cap0' with the class you want.
```
Partly class need to train about 2000 epoches.
### Train
You can eval:
```bash
# Y.ou can evaluate and change the evaluation setting depending on `config/eval_config`.
python eval.py
```
We will provide pre-trained weights after acceptance.

## Intraclass-Variance Benchmark
This is a dataset with intraclass variance called Anomaly-ShapeNet-Intraclass-Variance. This data contains 8 classes in both PCD and OBJ formats. The training dataset contains two normal samples each of two subcategories under the same broad category (four in total). The test dataset contains normal samples and multiple anomalies in both subcategories. Here are the **Statistical Data.**
The industrial test dataset used will be released upon acceptance, taking into account copyright reasons.

**Statistics from Anomaly-ShapeNet and AnomalyShape-Net-New , datasets organized by the authors.**
| class       | subcategories | template | positive | hole | bulge | concavity | crak | broken | scratch | bending | total |
|-------------|---------------|----------|----------|------|-------|-----------|------|--------|---------|---------|-------|
| bottle      | 2             | 4        | 30       | 2    | 14    | 14        | 1    | 2      | \       | \       | 67    |
| bowl        | 2             | 4        | 30       | \    | 14    | 14        | \    | \      | 8       | \       | 70    |
| bucket      | 2             | 4        | 30       | 4    | 14    | 14        | 4    | 4      | 2       | \       | 76    |
| cap         | 2             | 4        | 30       | 4    | 14    | 14        | \    | 4      | \       | 1       | 71    |
| cup         | 2             | 4        | 30       | \    | 14    | 14        | \    | \      | \       | \       | 62    |
| helmet      | 2             | 4        | 30       | 2    | 14    | 14        | 2    | 2      | 1       | 2       | 71    |
| microphone  | 2             | 4        | 30       | \    | 14    | 14        | \    | \      | \       | \       | 62    |
| tap         | 2             | 4        | 30       | 4    | 14    | 14        | 2    | 4      | 2       | \       | 74    |

**Statistical Data of the Data Made by the Author.**
| class         | subcategories | template | positive | missing | bulge | crack | twist | total |
|---------------|---------------|----------|----------|---------|-------|-------|-------|-------|
| cone          | 4             | 4        | 25       | 11      | 7     | 4     | 8     | 59    |
| bowl          | 4             | 4        | 25       | 10      | 7     | 3     | 7     | 56    |
| door          | 4             | 4        | 21       | 2       | 4     | 3     | 7     | 41    |
| keyboard      | 4             | 4        | 24       | 11      | 8     | 4     | 6     | 57    |
| night_stand   | 4             | 4        | 31       | 5       | 6     | 3     | 6     | 55    |
| radio         | 4             | 4        | 25       | 8       | 7     | 4     | 7     | 55    |
| vase          | 4             | 4        | 32       | 12      | 8     | 4     | 8     | 68    |
| xbox          | 4             | 4        | 29       | 11      | 8     | 3     | 8     | 63    |

# Acknowledge
This code partly from [PO3AD](https://arxiv.org/abs/2412.12617) [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) and [ISMP](https://arxiv.org/abs/2412.13461). If possible, please cite them.
```bibtex
@misc{ye2024po3adpredictingpointoffsets,
      title={PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection}, 
      author={Jianan Ye and Weiguang Zhao and Xi Yang and Guangliang Cheng and Kaizhu Huang},
      year={2024},
      eprint={2412.12617},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.12617}, 
}

@InProceedings{Li_2024_CVPR,
    author    = {Li, Wenqiao and Xu, Xiaohao and Gu, Yao and Zheng, Bozhong and Gao, Shenghua and Wu, Yingna},
    title     = {Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {22207-22216}
}

@misc{liang2025lookinsidemoreinternal,
      title={Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection}, 
      author={Hanzhe Liang and Guoyang Xie and Chengbin Hou and Bingshu Wang and Can Gao and Jinbao Wang},
      year={2025},
      eprint={2412.13461},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13461}, 
}
```
