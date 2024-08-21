### data_loaders.py
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

# Env
from networks import define_net
from utils import getCleanAllDataset
import torch
from torchvision import transforms
from options import parse_gpuids
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
import time


def merge_data(data_path, clinical_data_path):
    clinical_data_path = os.path.join(data_path, clinical_data_path)
    mirna_data_path = os.path.join(data_path, 'miRNA.csv')
    cnv_data_path = os.path.join(data_path, 'cnv.csv')
    clinical_df = pd.read_csv(clinical_data_path, sep='\t')
    clinical_df.rename(columns={'submitter_id':'SLIDE_ID'},inplace=True)
    mirna_df = pd.read_csv(mirna_data_path)
    cnv_df = pd.read_csv(cnv_data_path)
    
    merged_df = clinical_df[['SLIDE_ID','time','event']].merge(mirna_df, on='SLIDE_ID', how='inner') 
    merged_df = merged_df.merge(cnv_df, on='SLIDE_ID', how='inner')
    
    merged_df.to_csv('merged_patient_data.csv', index=False)
    merged_df.rename(columns={'submitter_id':'SLIDE_ID'},inplace=True)
    merged_df.rename(columns={'time':'survival_months'},inplace=True)
    return merged_df

def get_vgg_features(model, device, img_path):
    if model is None:
        return img_path 
    else:
        x_path = Image.open(img_path).convert('RGB')
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x_path = torch.unsqueeze(normalize(x_path), dim=0)
        features, hazard = model(x_path=x_path.to(device))
        return features.cpu().detach().numpy()
    
def get_vit_features(feat_path,model,device):
    
    x = torch.load(feat_path).to(device)
    features,hazard = model(x) #pass feature matrix to path model and return desired output.
    return features.cpu().detach().numpy() 

class custom_PathNet(nn.Module):

    def __init__(self, path_dim=32, act=None, num_classes=1):
        super(custom_PathNet, self).__init__()
       
        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classifier = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(512, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

    def forward(self, features):
        x = features
        x = torch.unsqueeze(x,dim=0)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        hazard = self.linear(features)

        return features, hazard


def getAlignedMultimodalData(opt, model, device, all_dataset, pat_split, pat2img):
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []
    for pat_name in pat_split:
        if pat_name not in all_dataset.values or pat_name not in pat2img.keys() : 
            print('skipped patient:', pat_name) 
            continue
        for img_fname in pat2img[pat_name]:
            
            start = time.time()
            print('preparing patient:', pat_name)
            img_fname = img_fname.replace('.h5','')
            path_fname = img_fname + '.pt'
            x_patname.append(str(all_dataset.loc[all_dataset['SLIDE_ID']==pat_name,'SLIDE_ID'].values))
            vit_f = os.path.join(opt.dataroot, opt.features, 'pt_files',path_fname)
            vit_f = get_vit_features(feat_path=vit_f,model=model,device=device)
            x_path.append(vit_f)
            x_omic.append(np.array(all_dataset[all_dataset['SLIDE_ID'] == pat_name].drop(metadata, axis=1)))    
            e.append(int(float(all_dataset[all_dataset['SLIDE_ID']==pat_name]['event'].values)))
            t.append(int(float(all_dataset[all_dataset['SLIDE_ID']==pat_name]['survival_months'].values)))
            end = time.time()
            print('elapsed time:', end-start)
    
    return x_patname, x_path, x_grph, x_omic, e, t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_name', type=str, default=None, help="name of the split set") # change this everytime
    parser.add_argument('--split_type', type=str, default=None, help="type of the split, options: [clinical | omic]")
    parser.add_argument('--clinical_data', type=str, default=None, help="clinical data csv/tsv file name")
    parser.add_argument('--pnas_split', type=str, default=None, help="pnas split path")
    parser.add_argument('--split_number', type=int, default=None, help="number of split")
    parser.add_argument('--dataroot', type=str, default='/media/nfs/TCGA_SLIDES/', help="datasets")
    parser.add_argument('--features', type=str, default=None, help="Path to slide features")
    parser.add_argument('--roi_dir', type=str, default='all_st')
    parser.add_argument('--graph_feat_type', type=str, default='.pt', help="graph features to use")
    parser.add_argument('--ignore_missing_moltype', type=int, default=0, help="Ignore data points with missing molecular subtype")
    parser.add_argument('--ignore_missing_histype', type=int, default=0, help="Ignore data points with missign histology subtype")
    parser.add_argument('--make_all_train', type=int, default=0)
    parser.add_argument('--use_vgg_features', type=int, default=0)
    parser.add_argument('--use_vit_features',type = int, default=1,help = "use wsi features from pre trained VIT model")
    parser.add_argument('--use_rnaseq', type=int, default=0)
    parser.add_argument('--checkpoints_dir', type=str, default='/media/nfs/immuno_project/model', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='immuno_15folds', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='custom', help='mode')
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')

    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    return opt

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
clinical_data_path = opt.clinical_data
if opt.split_type == 'clinical':
    all_dataset = pd.read_csv(os.path.join(opt.dataroot,clinical_data_path),sep='\t')
    all_dataset.rename(columns={'submitter_id':'SLIDE_ID'},inplace=True)
    all_dataset.rename(columns={'time':'survival_months'},inplace=True)
    #breakpoint()
elif opt.split_type == 'omic':
    all_dataset = merge_data(opt.dataroot, clinical_data_path)  
else:
    raise Exception("invalid split type !")

metadata = ['SLIDE_ID', 'survival_months', 'event']
### Creates a mapping from SLIDE_ID -> Image ROI
img_fnames = [f for f in os.listdir(os.path.join(opt.dataroot,'patches/patches')) if f.endswith('.h5')]
pat2img = {}

"""
This code also support only multiple WSI image per patient.
uncomment lines 178-181 for multiple WSI and comment 173-176
"""

#maps only one WSI img to patient
for img_fname in img_fnames:
    pat = img_fname[:12]  # Extract patient name from the filename
    pat2img[pat] = [img_fname]  # Map patient to image filename directly

# #maps multiple wsi img to patient
# for pat, img_fname in zip([img_fname[:12] for img_fname in img_fnames], img_fnames):
#     if pat not in pat2img.keys(): pat2img[pat] = []
#     pat2img[pat].append(img_fname) # maps multiple images to one patient if avalaible
  
### Dictionary file containing split information
data_dict = {}
data_dict['data_pd'] = all_dataset  
data_dict['pat2img'] = pat2img
data_dict['img_fnames'] = img_fnames
cv_splits = {}

### Extracting K-Fold Splits
pnas_splits = pd.read_csv(opt.dataroot+opt.pnas_split)
index = pnas_splits.index.name 
pnas_splits.columns = [index]+[str(k) for k in range(1, opt.split_number+1)]
pnas_splits.index = pnas_splits[index]
pnas_splits = pnas_splits.drop([index], axis=1)
pnas_splits.head()
### get path_feats
if opt.use_vit_features:
    model = custom_PathNet()
    model.eval()
    model.to(device)
    print('loaded path model')
else: 
    model = None
print(all_dataset.shape)
patient_counts_train= []
patient_counts_test = []
for k in pnas_splits.columns:
    print('Creating Split %s' % k)
    pat_train = pnas_splits.index[pnas_splits[k] == 'Train'] if opt.make_all_train == 0 else pnas_splits.index
    pat_test = pnas_splits.index[pnas_splits[k] == 'Test']
    cv_splits[int(k)] = {}

    train_x_patname, train_x_path, train_x_grph, train_x_omic, train_e, train_t = getAlignedMultimodalData(opt, model, device, all_dataset, pat_train, pat2img)
    test_x_patname, test_x_path, test_x_grph, test_x_omic, test_e, test_t = getAlignedMultimodalData(opt, model, device, all_dataset, pat_test, pat2img)
    train_x_omic, train_e, train_t = np.array(train_x_omic).squeeze(axis=1), np.array(train_e, dtype=np.float64), np.array(train_t, dtype=np.float64)
    test_x_omic, test_e, test_t = np.array(test_x_omic).squeeze(axis=1), np.array(test_e, dtype=np.float64), np.array(test_t, dtype=np.float64)
    patient_counts_train.append(len(train_x_omic))
    patient_counts_test.append(len(test_x_omic))
    
    if opt.split_type == 'clinical':
        scaler = preprocessing.StandardScaler().fit(train_x_omic) 
        train_x_omic = scaler.transform(train_x_omic)
        test_x_omic = scaler.transform(test_x_omic)
    else:
        train_x_omic =  OrdinalEncoder(categories = 'auto').fit_transform(train_x_omic) 
        test_x_omic =  OrdinalEncoder(categories = 'auto').fit_transform(test_x_omic)
    
    train_data = {'x_patname': train_x_patname,
                  'x_path':train_x_path, 
                  'x_grph':train_x_grph, 
                  'x_omic':train_x_omic, 
                  'e':np.array(train_e, dtype=np.float64), 
                  't':np.array(train_t, dtype=np.float64),
                  }

    test_data = {'x_patname': test_x_patname,
                 'x_path':test_x_path, 
                 'x_grph':test_x_grph, 
                 'x_omic':test_x_omic,
                 'e':np.array(test_e, dtype=np.float64),
                 't':np.array(test_t, dtype=np.float64),
                 }

    dataset = {'train':train_data, 'test':test_data}
    cv_splits[int(k)] = dataset

    if opt.make_all_train: break

    torch.cuda.empty_cache()
    
print('patient train samples on each split:',patient_counts_train)
print('patient test samples on each split:',patient_counts_test) 
data_dict['cv_splits'] = cv_splits
pickle.dump(data_dict, open(f'/media/nfs/TCGA_SLIDES/splits/{opt.split_name}.pkl', 'wb'))
