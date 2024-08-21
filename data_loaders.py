### data_loaders.py
import os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset 
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler


def normalize(omic_data):
    """
    accepts a np array and return min-max normalized np array
    """
    min_values = np.min(omic_data, axis=0)
    max_values = np.max(omic_data, axis=0)
    range_nonzero = max_values - min_values
    range_nonzero[range_nonzero == 0] = 1
    omic_data_scaled = (omic_data - min_values) / range_nonzero
    return omic_data_scaled


class VitkanDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.mode = mode
        self.omics_dims = []
        self.omics_dims.append(None)
        self.omics_dims.append(None)  
        self.omics_dims.append(self.X_omic.shape[1])
        self.opt = opt
        #self.X_omic_normalized = normalize(self.X_omic)
        if opt.net_VAE == 'conv_1d':
          self.X_omic_normalized = self.X_omic_normalized[np.newaxis, :, :]
        
        if opt.transform == 'standart_scaler':
            self.transform = StandardScaler()
        elif opt.transform == 'robust':
            self.transform = RobustScaler()
        elif opt.transform == 'normalize':
            self.transform = MinMaxScaler()
        else: self.transform = None
            
    def __getitem__(self, index):

        
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)

        if self.mode == 'custom' or self.mode == 'SingleVisionNet' or self.mode == 'DoubleFusionNet' or self.mode== 'SingleVisionNet_KAN':
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)           
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor).squeeze(0)

                 
            if self.opt.omic_model == 'vae':
                if self.opt.add_channel:
                   C_tensor =  torch.tensor(self.X_omic_normalized[:,index,:]).type(torch.FloatTensor)
                else:               
                    C_tensor = torch.tensor(self.X_omic_normalized[index]).type(torch.FloatTensor)
                A_tensor = 0
                if self.opt.ch_separate:
                    B_tensor = list(np.zeros(23))
                else:
                    B_tensor = 0
                omics_dict = {'input_omics': [A_tensor, B_tensor, C_tensor], 'index': index}
            else: omics_dict = {}
   
            return (single_X_path,  single_X_omic, single_e, single_t, omics_dict)

    def __len__(self):
        return len(self.X_path)
