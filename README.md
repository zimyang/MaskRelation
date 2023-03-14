# Mask Relation
The code of Masked Relation Learning for DeepFake Detection (TIFS 2023)

We provide Pytorch and MindSpore (昇思) versions of source code.

### Install 
- Python 3.8 
- Pytorch 1.12 
- MindSpore 1.10.1
- CUDA 11.1

### Train
```
python train.py --gpu [gpu_id] --multiprocessing_distributed True
```

### Test
```
python test.py --resume [path/to/checkpoint]
```
