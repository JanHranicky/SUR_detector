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
from PIL import Image,ImageOps

import matplotlib.pyplot as plt 
import argparse

TARGET_ID = "m421" #id of target person 
NUM_OF_EPOCHS = 30 #num of epochs for training
BATCH_SIZE = 32 #defined batch size used in training

class SurImageDataSet(Dataset):
    """Class implements Torch.Dataset class 
    Used in training 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self,features,labels):
        print(f'features len {len(features)}')
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SUR_Image_Net(nn.Module):
    def __init__(self):
        super(SUR_Image_Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,8,(3,3),stride=(1,1),padding=(0,0))
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d((2,2),stride=(2,2),padding=(0,0))

        self.conv2 = nn.Conv2d(8,16,(3,3),stride=(1,1),padding=(0,0))
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d((2,2),stride=(2,2),padding=(0,0))

        self.conv3 = nn.Conv2d(16,32,(3,3),stride=(1,1),padding=(0,0))
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d((2,2),stride=(2,2),padding=(0,0))

        self.conv4 = nn.Conv2d(32,64,(3,3),stride=(1,1),padding=(1,1))
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.maxpool4 = nn.MaxPool2d((2,2),stride=(2,2),padding=(1,1))

        self.conv5 = nn.Conv2d(64,128,(3,3),stride=(1,1),padding=(1,1))
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.maxpool5 = nn.MaxPool2d((2,2),stride=(2,2),padding=(1,1))

        self.conv6 = nn.Conv2d(128,256,(3,3),stride=(1,1),padding=(1,1))
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.maxpool6 = nn.MaxPool2d((2,2),stride=(2,2),padding=(1,1))

        self.conv7 = nn.Conv2d(256,512,(3,3),stride=(1,1),padding=(1,1))
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.maxpool7 = nn.MaxPool2d((2,2),stride=(2,2),padding=(1,1))

        self.dropout1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2048,100)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100,50)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(50,2)

        self.out = nn.Softmax()

    def forward(self,x):    

        x = self.maxpool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.maxpool3(F.relu(self.batchnorm3(self.conv3(x))))
        x = self.maxpool4(F.relu(self.batchnorm4(self.conv4(x))))
        x = self.maxpool5(F.relu(self.batchnorm5(self.conv5(x))))
        x = self.maxpool6(F.relu(self.batchnorm6(self.conv6(x))))
        x = self.maxpool7(F.relu(self.batchnorm7(self.conv7(x))))

        x = torch.flatten(x,1)
        x = self.dropout1(x)

        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.dropout3(F.relu(self.fc2(x)))

        x = self.out(self.fc3(x))

        return x

def create_image_dataframe(folders):
    """Function that creates pandas dataframe constiting 
    of .png files inside specified directories

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
            speaker_id = str(png_files[i]).split('\\')[3].split('_')[0]
            speakers.append(1 if speaker_id == TARGET_ID else 0)
    df = pd.DataFrame(filelist)
    df = df.rename(columns={0:'file'})
    df['target'] = speakers
    return df

def convert_to_tensor(files):
    #print(files.file)
    img = Image.open(files.file).convert('RGB')
    gray_image = ImageOps.grayscale(img)
    convert_tensor = transforms.ToTensor()
    converted = convert_tensor(gray_image)
    return converted

def create_dataLoader(df,features,batch_size=32,shuff=True):
    """Creates instance of torch.utils.data.Dataloader
    from Pandas.Dataframe

    Args:
        df (Pandas.DataFrame): dataframe 
        features (np.array): np.array of features
        batch_size (int, optional): Batch size used in Dataloader. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader
    """
    ds = SurImageDataSet(features,df['target'])
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuff, pin_memory=True)

def create_nn(path=None):
    """Handles creation of NN. 
    Loads existing module, if path is given

    Args:
        path (str, optional): Path of existing NN model. Defaults to None.

    Returns:
        Torch.nn.Module: instance of neural network
    """
    sur_img_net = SUR_Image_Net()

    if path != None:
        sur_img_net.load_state_dict(torch.load("./models/"+path+".pt"))
        sur_img_net.eval()
    return sur_img_net

def train_nn(model,train_loader,val_loader,val_df,output_name = None):
    """Trains neural network on given Dataloader

    Args:
        model (Torch.nn.module): neural network to train
        train_loader (torch.utils.data.DataLoader): Dataloader used to train
        output_name (str, optional): If specified, model is saved to disk with given name after training. Defaults to None.
    """
    criterion = nn.CrossEntropyLoss() #binary cross entropy as loss function (classification into two classed, is_target/isnt_target)
    optimizer = optim.Adam(model.parameters(),lr=0.0005) #optimization algorithm used is Stochastic gradient descent

    running_loss = 0
    iter = 0

    for i in range(NUM_OF_EPOCHS): #train NN for specified number of epochs
        batch_num = 0 #batch number for info print
        for batch_data,batch_labels in train_loader: #iterate over all batches
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(batch_data)
            #outputs = torch.Tensor([torch.argmax(a) for a in outputs])
            #print(f'outputs {outputs}')
            #outputs = torch.reshape(outputs,(len(outputs),)) #reshape output to be 1D array 
            
            batch_labels = batch_labels.to(torch.long) 

            loss = criterion(outputs, batch_labels) #apply loss function
            loss.backward() #back propagate
            optimizer.step() 

            iter += 1
            running_loss += loss.item()
            val_acc = test_nn(model,val_loader,val_df)
            print(f'TRAINING [{i + 1}, {batch_num + 1:5d}] loss: {running_loss}. Validation acc = {val_acc}') #print training info
            running_loss = 0.0
            batch_num += 1


    if output_name != None: #if name was specified, save model
        torch.save(model.state_dict(), "./models/"+output_name+".pt")


def test_nn(model,test_loader,df,final_test=False):
    """Tests NN on given validation dataset. Prints accuracy to console

    Args:
        model (Torch.nn.module): NN model
        test_loader (torch.utils.data.DataLoader): DataLoader with validation data
    """
    all = 0
    correct = 0
    for batch_data, batch_labels in test_loader: #iterates over all validation data
        all += len(batch_data) #increase lenght of all data
        #print(f'batch data size {batch_data.size()}')
        #print(f'batch_labels {batch_labels}')
        #gets output and changes it to 0 if output[i] < 0.5 or 1 if output[i] = 0.5
        outputs = model(batch_data)
        #outputs = torch.reshape(outputs,(len(outputs),))
        outputs = outputs.detach().numpy()
        #print(f'outputs {outputs}')
        outputs = [np.argmax(a) for a in outputs]
        #print(f'outputs {outputs}')


        curr = 0
        for data in outputs: #iterates over all outputs and increments correct variable if output was correct
            if data == batch_labels[curr]:
                correct += 1
            #prints output and it's corrent label for info
            if final_test:
                print(f'[out,GT] = [{int(outputs[curr])},{batch_labels[curr]}]')
            curr += 1
        
    return correct/all

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

    df_train = create_image_dataframe(['../data/non_target_train','../data/target_train'])
    df_val = create_image_dataframe(['../data/non_target_dev','../data/target_dev'])
    train_features = df_train.apply(convert_to_tensor,axis=1)
    val_features = df_val.apply(convert_to_tensor,axis=1)
    
    train_dl = create_dataLoader(df_train,train_features,BATCH_SIZE)    
    val_dl = create_dataLoader(df_val,val_features,BATCH_SIZE,False)

    sur_img_net = create_nn(load)

    if mode == 't' or mode == 'tv':
        train_nn(sur_img_net,train_dl,val_dl,df_val,output_name=save)  
    if mode == 'v' or mode == 'tv': 
        test_acc = test_nn(sur_img_net,val_dl,df_val,final_test=True)
        print(f'Accuracy after training {test_acc}') #prints final accuracy

if __name__ == '__main__':
    main()