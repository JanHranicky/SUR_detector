import voice_nn as voice
import image_nn as image

import torch
from torch.utils.data import Dataset
import numpy as np

import sys
import argparse
from os.path import exists

class SurValidationDataset(Dataset):
    """Class implements Torch.Dataset class 
    Used for data validataion

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self,image_features,voice_features):
        self.image_features = image_features
        self.voice_features = voice_features

    def __len__(self):
        return len(self.image_features) #all lists are equal in lenght so it doesnt matter 

    def __getitem__(self, idx):
        return self.image_features[idx],self.voice_features[idx]

def create_dataLoader(image_features,voice_features,batch_size=32):
    """Creates instance of torch.utils.data.Dataloader
    from Pandas.Dataframe

    Args:
        df (Pandas.DataFrame): dataframe 
        features (np.array): np.array of features
        batch_size (int, optional): Batch size used in Dataloader. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader
    """
    ds = SurValidationDataset(image_features,voice_features.astype(np.float32))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, pin_memory=True)

def classify_voice(model,data):
    outputs = model(data)
    outputs = torch.reshape(outputs,(len(outputs),))
    outputs = outputs.detach().numpy()
    """
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    """
    return outputs

def classify_face(model,data):
    outputs = model(data)
    outputs = outputs.detach().numpy()
    outputs = outputs[:,[1]]
    outputs = np.squeeze(outputs)
    return outputs

def sum_rule(face_prob,voice_prob):
    NUM_OF_CLASSIFERS = 2
    APRIOR = 0.5
    return (1-NUM_OF_CLASSIFERS)*APRIOR+(face_prob+voice_prob)

def construct_result_str(file_name,score,decision):
    return file_name + " " + str(score) + " " + str(decision) + "\n"

def save_file(list,file_name):
    if len(list) > 0:
        with open("./out/"+file_name,'w') as f:
            for r in list:
                f.write(r)

def classify_input(ds_loader,image_model,voice_model,image_df_val,voice_df_val,mode):
    voice_result_list = []
    image_result_list = []
    both_result_list = []

    voice_cnt = 0
    img_cnt = 0
    both_cnt = 0
    print(f'validation mode {mode}')
    for image_batch_data,voice_batch_data in ds_loader: #iterates over all validation data
        out_name = ""
        classified_face = classify_face(image_model,image_batch_data)
        classified_voice = classify_voice(voice_model,voice_batch_data)

        if mode == 'v' or mode == 'all':
            out_name = "voice_out.txt"

            for cv in classified_voice:
                file_name = str(image_df_val['file'][voice_cnt]).split('\\')[3].split('.')[0]
                decision = 1 if cv >= 0.5 else 0
                voice_result_list.append(construct_result_str(file_name,cv,decision))
                
                voice_cnt += 1
        if mode == 'i' or mode == 'all':
            out_name = "image_out.txt"

            for cf in classified_face:
                file_name = str(image_df_val['file'][img_cnt]).split('\\')[3].split('.')[0]
                decision = 1 if cf >= 0.5 else 0
                image_result_list.append(construct_result_str(file_name,cf,decision))
                
                img_cnt += 1
        if mode == 'vi' or mode == 'all':
            out_name = "both_out.txt"

            target_prob = sum_rule(classified_face,classified_voice)
            for tp in target_prob:
                file_name = str(image_df_val['file'][both_cnt]).split('\\')[3].split('.')[0]
                decision = 1 if tp >= 0.5 else 0
                both_result_list.append(construct_result_str(file_name,tp,decision))

                both_cnt += 1

    save_file(voice_result_list,"voice_out.txt")
    save_file(image_result_list,"image_out.txt")
    save_file(both_result_list,"both_out.txt")

def parse_arguments():
    mode = "all"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['v', 'i', 'vi','all'],help='Changes mode of the script to one of following vi = validate on dataset on both classifiers, v = validate on dataset using only voice classifier, i = validate on dataset using only image classifier, (default option) all = validates all options [v,i,vi]')
    parser.add_argument('--image',help='Name of existing image model without extension',required=True)
    parser.add_argument('--voice',help='Name of existing voice model without extension',required=True)

    args = vars(parser.parse_args())
    
    if args['mode'] != None:
        mode = args['mode']

    return mode,args['image'],args['voice']

def main():
    mode,image_model_name,voice_model_name = parse_arguments()

    image_df_val = image.create_image_dataframe(['../data/eval'])
    image_val_features = image_df_val.apply(image.convert_to_tensor,axis=1)

    voice_df_val = voice.create_dataframe(['../data/eval'])
    
    file_exists = exists('./features/voice_val_features.npy')
    #computes features of .wav files and saves them in np array
    #this takes a while thats why we compute feature only once and save them
    if not file_exists:
        print(f'File with voice dataset features NOT detected. Extracting features, this might take few minutes.')
        voice_val_features = voice.handle_feature_extraction(voice_df_val)
        voice.handle_np_array_save('./features/voice_val_features',voice_val_features)
        print(f'Features extracted. Created "voice_val_features.npy" file.')
    else:
        print(f'File with voice dataset features detected.')
        voice_val_features = voice.load_features('./features/voice_val_features')
        print(f'Features loaded from "voice_val_features.npy" file')

    val_dl = create_dataLoader(image_val_features,voice_val_features)

    img_classifier = image.create_nn(image_model_name)
    voice_classifier = voice.create_nn(voice_model_name)
    
    classify_input(val_dl,img_classifier,voice_classifier,image_df_val,voice_df_val,mode)

    

if __name__ == '__main__':
    main()