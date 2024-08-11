import os,glob,pathlib
import pandas as pd
import numpy as np
import librosa

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt 

from torchvision.utils import save_image

import random

TARGET_ID = "m421" #id of target person 


def create_image_dataframe(folders):
    """Function that creates pandas dataframe constiting 
    of .wav files inside specified directories

    Args:
        folders (list): list of folders to look in

    Returns:
        Pandas.DataFrame: dataframe with filenames and speaker ids
    """
    filelist = []
    speakers = []
    for f in folders:
        png_files = list(pathlib.Path(f).glob('*.png'))
        filelist += png_files
        for i in range(0, len(png_files)):
            speaker_id = str(png_files[i]).split('\\')[1].split('_')[0]
            speakers.append(1 if speaker_id == TARGET_ID else 0)
    df = pd.DataFrame(filelist)
    df = df.rename(columns={0:'file'})
    df['target'] = speakers
    return df

def convert_to_tensor(files):
    #print(files.file)
    img = Image.open(files.file)

    convert_tensor = transforms.ToTensor()
    converted = convert_tensor(img)
    return torch.flip(converted,[-1])

def scale(img, p=0.45):
    if random.random() < p:
        crop = transforms.CenterCrop([65,65])
        scale= transforms.Resize([80,80])

        return scale(crop(img)),"scale"
    return img,""

def gaus(img, p=0.75):
    if random.random() < p:
        gauss_blur = transforms.GaussianBlur([7,7])

        return gauss_blur(img),"_gauss"
    return img,""

def main():
    folder = 'target_train'
    target_train_df = create_image_dataframe(['../data/target_train'])
    train_features = target_train_df.apply(convert_to_tensor,axis=1)

    for i in range(len(train_features)):
        img = train_features[i]
        transform_str = "_rotated"
        img,tran_str = scale(img)
        transform_str += tran_str

        img,tran_str = gaus(img)
        transform_str += tran_str

        file_id = str(target_train_df['file'][i]).split('\\')[3].split('.')[0]

        save_image(img,'../data/'+folder+'/'+file_id+transform_str+str(i)+'.png')
    #print(train_features)

if __name__ == '__main__':
    main()
