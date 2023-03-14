import json
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
import sklearn
import mindspore
from mindspore.dataset import GeneratorDataset as Dataset
from albumentations import Compose, Resize, HorizontalFlip, Normalize
# from mindspore.dataset.transforms import Compose
# from mindspore.dataset import vision
import numpy as np
# Global
seq_len = 100

def augmentation(type='augment0', resize=112):
    augment0 = Compose([Resize(resize, resize),HorizontalFlip(),Normalize(mean=(0.43216,0.394666,0.37645),std=(0.22803,0.22145,0.216989))])
    augment_test = Compose([Resize(resize, resize), Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    augment_set = {'augment0': augment0, 'augment_test': augment_test}
    return augment_set[type]

class DFDataset:
    def __init__(self, phase, datalabel, resize, tag, codec, augment='augment0'):
        assert phase in ['train', 'val', 'test']
        self.datalabel = datalabel
        self.resize = resize
        self.phase = phase
        self.epoch = 0
        self.len = 0
        self.fake = []
        self.real = []
        self.dataset = []
        self.aug = augmentation(augment, resize)
        
        if phase == 'train':
            print("------Train Set------")
        elif phase == 'val':
            print("------Validation Set------")
        elif phase == 'test':
            print("------Test Set------")
        else:
            print("Error: The phase is None")
        if 'ff-all' in self.datalabel:
            if tag == "":
                print("Load ff-all")
                for subtag in ['deepfakes', 'face2face', 'faceswap', 'neural_textures', 'original', 'FaceShifter']:
                    subdataset = FF_dataset(subtag, codec, phase)
                    self.dataset += subdataset
                    if len(subdataset) > 0:
                        print(f'load {subtag}-{codec} len: {len(subdataset)}')
            else:
                for subtag in ['deepfakes', 'face2face', 'faceswap', 'neural_textures', 'original']:
                    if tag != subtag:
                        subdataset = FF_dataset(subtag, codec, phase)
                        self.dataset += subdataset
                        if len(subdataset) > 0:
                            print(f'load {subtag}-{codec} len: {len(subdataset)}')
            if phase != 'test':
                self.dataset = make_balance(self.dataset)
        
        elif 'ff' in self.datalabel:
            self.dataset = FF_dataset(tag, codec, phase)
            self.dataset += FF_dataset("original", codec, phase)
            print(f'load {tag}-{codec} len: {len(self.dataset)}')
        
        elif 'celeb' in self.datalabel:
            self.dataset = CelebDF(phase)
            print(f'load {self.datalabel} len: {len(self.dataset)}')
        elif 'dfdc' in self.datalabel:
            self.dataset = DFDC(phase)
            print(f'load {self.datalabel} len: {len(self.dataset)}')

        else:
            raise(Exception(f'Error: Dataset {self.datalabel} does not exist!'))
        self.len = len(self.dataset)
        
    def __getitem__(self, index):
        fpath_list, label = self.dataset[index]
        
        # Total number of sampled frames.
        len_list = len(fpath_list)
        frame_N = len_list

        buffer = np.empty(shape=(seq_len, self.resize, self.resize, 3), dtype=np.float64)
        idx = 0
        for idx, i in enumerate(range(frame_N)):
            fpath = fpath_list[i]
            img = cv2.imread(fpath)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.aug(image=img)["image"]
            buffer[idx] = img
        idx += 1; cur_idx = 0
        while idx < seq_len:
            buffer[idx] = buffer[cur_idx % frame_N]
            cur_idx += 1; idx += 1
        buffer = self.ToTensor(buffer)
        return buffer, label
        # return {'img': buffer, 'label': label}
    
    def __len__(self):
        return self.len
    
    def ToTensor(self, ndarray):
        tensor = ndarray.transpose((3, 0, 1, 2)) 
        return mindspore.Tensor.from_numpy(tensor).astype(mindspore.float32)

def make_balance(data):
    tr = list(filter(lambda x:x[1]==0, data))
    tf = list(filter(lambda x:x[1]==1, data))
    if len(tr) > len(tf):
        tr, tf = tf, tr
    rate = len(tf) // len(tr)
    res = len(tf) - rate * len(tr)
    tr = tr * rate + random.sample(tr,res)
    return tr + tf

## Face Forensics++
ffpp_raw_train = "/data2/ziming.yang/datasets/ffpp/ffpp_raw_train.json"
ffpp_raw_val = "/data2/ziming.yang/datasets/ffpp/ffpp_raw_val.json"
ffpp_raw_test = "/data2/ziming.yang/datasets/ffpp/ffpp_raw_test.json"
ffpp_c23_train = "/data2/ziming.yang/datasets/ffpp/ffpp_c23_train.json"
ffpp_c23_val = "/data2/ziming.yang/datasets/ffpp/ffpp_c23_val.json"
ffpp_c23_test = "/data2/ziming.yang/datasets/ffpp/ffpp_c23_test.json"
ffpp_c40_train = "/data2/ziming.yang/datasets/ffpp/ffpp_c40_train.json"
ffpp_c40_val = "/data2/ziming.yang/datasets/ffpp/ffpp_c40_val.json"
ffpp_c40_test = "/data2/ziming.yang/datasets/ffpp/ffpp_c40_test.json"

ffpp_proto = {"raw": {'train': ffpp_raw_train, 'val': ffpp_raw_val, 'test': ffpp_raw_test},
"c23": {'train': ffpp_c23_train, 'val': ffpp_c23_val, 'test': ffpp_c23_test},
"c40": {'train': ffpp_c40_train, 'val': ffpp_c40_val, 'test': ffpp_c40_test}
}
def FF_dataset(tag, codec, phase='train'):
    assert(tag in ['deepfakes', 'face2face', 'faceswap', 'neural_textures', 'original', 'FaceShifter'])
    assert(codec in ['raw','c23','c40','all'])
    assert(phase in ['train','val','test','all'])
    if phase=="all":
        return FF_dataset(tag, codec, 'train') + FF_dataset(tag, codec, 'val') + FF_dataset(tag, codec, 'test')
    if codec=="all":
        return FF_dataset(tag,'raw', phase) + FF_dataset(tag,'c23',phase) + FF_dataset(tag,'c40',phase)
    
    _dataset = []
    path = ffpp_proto[codec][phase]
    with open(path, 'r') as f:
        data_dict = json.load(f)
    for k,v in data_dict.items():
        if tag not in k:
            continue
        video_label = v['label']
        file_list = v['list']
        
        for i in range(0, len(file_list), seq_len):
            if i+seq_len>=len(file_list):
                _dataset.append([file_list[-seq_len:], video_label])
            else:
                _dataset.append([file_list[i:i+ seq_len], video_label])
    return _dataset

## Celeb-DF v2
celebtrain = "/data2/ziming.yang/datasets/Celeb-DF/celeb_train.json"
celebtest = "/data2/ziming.yang/datasets/Celeb-DF/celeb_test.json"
celeb_proto = {'train': celebtrain, 'test': celebtest}
def CelebDF(phase='train'):
    assert(phase in ['train', 'test', 'all'])
    if phase=='all':
        return CelebDF('train') + CelebDF('tests')
    _dataset = []
    path = celeb_proto[phase]
    with open(path, 'r') as f:
        data_dict = json.load(f)
    num_frames = 100 if phase!='train' else None
    for k,v in data_dict.items():
        if len(v['list']) != 30: ## Dataset analysis
            continue
        video_label = v['label']
        file_list = v['list'][:num_frames]

        len_list = len(file_list)
        for i in range(0, len_list, seq_len):
            if i+seq_len >= len_list:
                _dataset.append([file_list[-seq_len:], video_label])
            else:
                _dataset.append([file_list[i:i+seq_len], video_label])

    return _dataset

## Deepfakes Detection Challenge
dfdctrain = "/data2/ziming.yang/datasets/DFDC/dfdc_train.json"
dfdcval = "/data2/ziming.yang/datasets/DFDC/dfdc_val.json"
dfdctest = "/data2/ziming.yang/datasets/DFDC/dfdc_test.json"
dfdc_proto = {'train': dfdctrain, 'val': dfdcval, 'test': dfdctest}
def DFDC(phase='train'):
    assert(phase in ['train', 'val', 'test', 'all'])
    if phase=='all':
        return DFDC('train') + DFDC('val') + DFDC('tests')
    _dataset = []
    path = dfdc_proto[phase]
    num_frames = 100 if phase!='train' else None
    with open(path, 'r') as f:
        data_dict = json.load(f)
    for k,v in data_dict.items():
        if len(v['list']) < 100:
            continue

        video_label = v['label']
        
        file_list = v['list'][:num_frames]
        
        len_list = len(file_list)

        for i in range(0, len_list, seq_len):
            if i+seq_len >= len_list:
                _dataset.append([file_list[-seq_len:], video_label])
            else:
                _dataset.append([file_list[i:i+ seq_len], video_label])
    return _dataset