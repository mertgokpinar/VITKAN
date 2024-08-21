import os
import logging
import numpy as np
import random
import pickle

import torch
import wandb
import yaml
# Env
from data_loaders import *
from options import parse_args
from train_test import train, test


### 1. Initializes parser and device
opt = parse_args()

if opt.wandb:
	with open("./config.yaml") as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
	run = wandb.init(config=config)
	wandb.config.update(opt)

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
#device = torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir) #makes checkpoint directory -mg
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name)) 
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
# ignore_missing_histype = 1 if 'grad' in opt.task else 0
# ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
# use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
# use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''
# data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)

data_cv_path = os.path.join(opt.dataroot, opt.split_directory)

print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []
train_cindex =[]

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
	print("*******************************************")
	if os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d_patch_pred_train.pkl' % (opt.model_name, k))):
		print("Train-Test Split already made.")
		continue
	### 3.1 Trains Model
	model, optimizer, metric_logger, omic_model = train(opt, data, device, k)

	#breakpoint()
	### 3.2 Evalutes Train + Test Error, and Saves Model
	loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train, vae_loss = test(opt, model, data, 'train', device, omic_model)
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, vae_loss = test(opt, model, data, 'test', device, omic_model)
 
	if opt.wandb:
		wandb.log(
		{
			"split": k,
			"loss_train": loss_train,
			"loss_test": loss_test,
			"cindex_train": cindex_train,
			"cindex_test": cindex_test,
		}
		)
	
	if opt.task == 'surv':
		print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results.append(cindex_test)
		train_cindex.append(cindex_train)

	### 3.3 Saves Model
	model_state_dict = model.cpu().state_dict()
	torch.save({
		'split':k,
	    'opt': opt,
	    'epoch': opt.niter+opt.niter_decay,
	    'data': data,
	    'model_state_dict': model_state_dict,
	    'optimizer_state_dict': optimizer.state_dict(),
	    'metrics': metric_logger}, 
	    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

	print()

	pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
	pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))


print('Split Results:', results)
print("Average:", np.array(results).mean())
if opt.wandb:
	wandb.log({
		'Final avarage c-index': np.array(results).mean(),
		'avarage train c-index': np.array(cindex_train).mean()
	})
pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))