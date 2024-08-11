from ast import arg
import os,glob,pathlib
import pandas as pd
import numpy as np
import librosa

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import matplotlib.pyplot as plt 

import argparse
from os.path import exists


TARGET_ID = "m421" #id of target person 
NUM_OF_EPOCHS = 300 #num of epochs for training
BATCH_SIZE = 32 #defined batch size used in training

class SurVoiceDataset(Dataset):
    """Class implements Torch.Dataset class 
    Used in training 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SUR_Net(nn.Module):
    """Definition of NN class used for detection 
    of one person based on their voice. 
    It constists of two 1D convolutional layers,
    each with their MaxPool layers, and 3 fully connected layers
    Ouput activation is sigmoid in order to get probability value.
    Which is then interpret as follows: p >= 0.5 -> person is detected. 
    """
    def __init__(self):
        super(SUR_Net, self).__init__()
        
        self.conv1 = nn.Conv1d(1,3,3)
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(3,6,3)

        self.fc1 = nn.Linear(276,128)
        self.dropout1 = nn.Dropout(p=0.05)

        self.fc2 = nn.Linear(128,128)
        self.dropout2 = nn.Dropout(p=0.025)

        self.fc3 = nn.Linear(128,128)
        self.dropout3 = nn.Dropout(p=0.012)

        self.fc4 = nn.Linear(128,1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x[:, None, :]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        #print(x.size())
        x = torch.flatten(x,1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.out(self.fc4(x))
        return x

def extract_features(files):
    """Function that extracts features from .wav file
    Function is adopted from https://towardsdatascience.com/how-to-build-a-neural-network-for-voice-classification-5e2810fe1efa
    voice classification course by Jurgen Arias

    Args:
        files (str): String name of directory with .wav files

    Returns:
        mfccs, chroma, mel, contrast, tonnetz: features of .wav file
    """
    # Sets the name to be the path to where the file is in my computer
    file_name = files.file
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def create_dataframe(folders):
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
        wav_files = list(pathlib.Path(f).glob('*.wav'))
        filelist += wav_files
        for i in range(0, len(wav_files)):
            speaker_id = str(wav_files[i]).split('\\')[3].split('_')[0]
            speakers.append(1 if speaker_id == TARGET_ID else 0)
    df = pd.DataFrame(filelist)
    df = df.rename(columns={0:'file'})
    df['speaker'] = speakers
    return df

def handle_feature_extraction(df):
    """Extracs featues from all files specified in dataframe

    Args:
        df (Pandas.DataFrame): dataframe with file locations

    Returns:
        np.array: array of features
    """
    train_features = df.apply(extract_features,axis=1)
    features_train = []
    for i in range(0, len(train_features)):
        features_train.append(np.concatenate((
            train_features[i][0],
            train_features[i][1], 
            train_features[i][2], 
            train_features[i][3],
            train_features[i][4]), axis=0))

    return np.array(features_train)


def handle_np_array_save(file_name,arr):
    """Helper functions that saves np.array to disk
    Args:
        file_name (str): name of output file
        arr (np.array): numpy array to save
    """
    with open(file_name+'.npy', 'wb') as f:
        np.save(f, arr)

def load_features(file_name):
    """Helper function that loads np.array from disk
    Args:
        file_name (str): file name 

    Returns:
        np.array
    """
    with open(file_name+'.npy', 'rb') as f:
        features = np.load(f)
    return features

def create_dataLoader(df,features,batch_size=32):
    """Creates instance of torch.utils.data.Dataloader
    from Pandas.Dataframe

    Args:
        df (Pandas.DataFrame): dataframe 
        features (np.array): np.array of features
        batch_size (int, optional): Batch size used in Dataloader. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader
    """
    ds = SurVoiceDataset(features.astype(np.float32),df['speaker'])
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True)

def create_nn(path=None):
    """Handles creation of NN. 
    Loads existing module, if path is given

    Args:
        path (str, optional): Path of existing NN model. Defaults to None.

    Returns:
        Torch.nn.Module: instance of neural network
    """
    sur_net = SUR_Net()

    if path != None:
        sur_net.load_state_dict(torch.load("./models/"+path+".pt"))
        sur_net.eval()
    return sur_net

def train_nn(model,train_loader,output_name = None):
    """Trains neural network on given Dataloader

    Args:
        model (Torch.nn.module): neural network to train
        train_loader (torch.utils.data.DataLoader): Dataloader used to train
        output_name (str, optional): If specified, model is saved to disk with given name after training. Defaults to None.
    """
    criterion = nn.BCELoss() #binary cross entropy as loss function (classification into two classed, is_target/isnt_target)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #optimization algorithm used is Stochastic gradient descent

    running_loss = 0
    iter = 0

    for i in range(NUM_OF_EPOCHS): #train NN for specified number of epochs
        batch_num = 0 #batch number for info print
        for batch_data,batch_labels in train_loader: #iterate over all batches

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(batch_data)
            outputs = torch.reshape(outputs,(len(outputs),)) #reshape output to be 1D array 
            
            batch_labels = batch_labels.to(torch.float32) 

            loss = criterion(outputs, batch_labels) #apply loss function
            loss.backward() #back propagate
            optimizer.step() 

            iter += 1
            running_loss += loss.item()
            print(f'[{i + 1}, {batch_num + 1:5d}] loss: {running_loss}') #print training info
            running_loss = 0.0
            batch_num += 1

    if output_name != None: #if name was specified, save model
        torch.save(model.state_dict(), "./models/"+output_name+".pt")


def test_nn(model,test_loader):
    """Tests NN on given validation dataset. Prints accuracy to console

    Args:
        model (Torch.nn.module): NN model
        test_loader (torch.utils.data.DataLoader): DataLoader with validation data
    """
    all = 0
    correct = 0
    for batch_data, batch_labels in test_loader: #iterates over all validation data
        all += len(batch_data) #increase lenght of all data

        #gets output and changes it to 0 if output[i] < 0.5 or 1 if output[i] = 0.5
        outputs = model(batch_data)
        outputs = torch.reshape(outputs,(len(outputs),))
        outputs = outputs.detach().numpy()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0

        curr = 0 
        for data in outputs: #iterates over all outputs and increments correct variable if output was correct
            if data == batch_labels[curr]:
                correct += 1
            curr += 1

        #prints output and it's corrent label for info
        print(f'[out,GT] = [{int(outputs[0])},{batch_labels[0]}]')

    print(f'Accuracy {correct/all}') #prints final accuracy

def parse_arguments():
    mode = "tv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tv', 't', 'v'],help='Changes mode of the script to one of following (default) tv = train and validate, t = train only, v = validate only')
    parser.add_argument('--loadmodel',help='Name of existing model without extension. If provided loads model with its weights')
    parser.add_argument('--savemodel',help='Name of model to save without extension. If provided saves model with givne name')

    args = vars(parser.parse_args())
    
    if args['mode'] != None:
        mode = args['mode']
    
    if args['mode'] == 'v' and args['loadmodel'] == None:
        print(f'You have to provide existing model for testing. Please run script with --loadmodel parameter.')
        exit(-1)

    return mode,args['loadmodel'],args['savemodel']

def main():
    mode,load,save = parse_arguments()

    df_train = create_dataframe(['../data/non_target_train','../data/target_train'])
    df_val = create_dataframe(['../data/non_target_dev','../data/target_dev'])
    
    train_exists = exists('./features/train_all.npy')
    val_exists = exists('./features/val_all.npy')

    if not train_exists:
        print(f'File with train voice dataset features NOT detected. Extracting features, this might take few minutes.')
        train_features = handle_feature_extraction(df_train)
        handle_np_array_save('./features/train_all',train_features)
        print(f'Features extracted. Created "train_all.npy" file.')
    else:
        print(f'File with voice train dataset features detected.')
        train_features = load_features('./features/train_all')
        print(f'Features loaded from "train_all.npy" file')

    if not val_exists:
        print(f'File with validate voice dataset features NOT detected. Extracting features, this might take few minutes.')
        val_features = handle_feature_extraction(df_val)
        handle_np_array_save('./features/val_all',val_features)
        print(f'Features extracted. Created "val_all.npy" file.')
    else:
        print(f'File with voice validation dataset features detected.')
        val_features = load_features('./features/val_all')
        print(f'Features loaded from "val_all.npy" file')

    train_dl = create_dataLoader(df_train,train_features,BATCH_SIZE)
    val_dl = create_dataLoader(df_val,val_features,1)
    #all_dl = create_dataLoader(df_all,X_all,BATCH_SIZE)

    sur_net = create_nn(load)

    if mode == 't' or mode == 'tv':
        train_nn(sur_net,train_dl,save)
    if mode == 'v' or mode == 'tv':    
        test_nn(sur_net,val_dl)

if __name__ == '__main__':
    main()