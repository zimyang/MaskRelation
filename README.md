# Mask Relation
The code of [Masked Relation Learning for DeepFake Detection](https://ieeexplore.ieee.org/document/10054130) (TIFS 2023).

We provide Pytorch and MindSpore (昇思) versions of source code.

### Install 
- Python 3.8 
- Pytorch 1.12 
- MindSpore 1.10.1
- CUDA 11.1

### Requirements
```
pip install albumentations --user
pip install pytorchcv --user
pip install timm --user
pip install einops --user
```


### Train
```
python train_ffpp.py --gpu 0 --amp [--multiprocessing_distributed True]
```

### Test
```
python test.py --resume [path/to/checkpoint]
```
