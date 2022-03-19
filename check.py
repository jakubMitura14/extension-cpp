from __future__ import division
from __future__ import print_function

import numpy as np

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,

)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

from monai.transforms import *


from monai.transforms import (
    LoadImage, LoadImaged, EnsureChannelFirstd,
    Resized,  Compose
)
from monai.config import print_config
import re






import argparse
import numpy as np
import torch



from torch.utils.cpp_extension import load
lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)



import lltm_baseline
import lltm

import lltm_cuda
device = torch.device("cuda")


def benchmarkMitura():
    set_determinism(seed=0)
    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


    data_dir = "D:\\dataSets\\CTORGmini\\"

    train_images = sorted(
        glob.glob(os.path.join(data_dir, "volumes 0-49", "*.nii.gz")))

    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    check_ds = Dataset(data=data_dicts, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    print(" cccccccccccccc\n  ")


    for dat in check_loader:
        print("**********************   \n  ")
        sizz = dat['image'].shape
        
        labelBoolTensorA =  torch.flatten(torch.where( dat['label']==1, 1, 0).bool().to(device))
        summA= torch.sum(labelBoolTensorA)
        labelBoolTensorB =torch.where( dat['label']==2, 1, 0).bool().to(device)
        summB= torch.flatten(torch.sum(labelBoolTensorB))
        print(summA )
        print(" \n  ")
        print(summB )
        print("   \n  ")
        print(sizz[2] )
        print(sizz[3] )
        print(sizz[4] )



        lltm_cuda.forwardB(labelBoolTensorA, labelBoolTensorB,sizz[2], sizz[3],sizz[4])
        print("after function in loop   \n  ")


benchmarkMitura()












#from torch.utils.cpp_extension import load
#lltm_cuda = load(
#    'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
#help(lltm_cuda)





#kwargs = {'dtype': torch.float64,
#          'device': device,
#          'requires_grad': True}
#X = torch.ones(5, dtype= torch.int32).to(device)
#Y = torch.ones(5,dtype= torch.int32).to(device)
#lltm_cuda.forwardB(X, Y)

##check_forward(variables, True, True)

#print("x = {}".format(Y.flatten()))