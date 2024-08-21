import os
import logging
import numpy as np
import random
import pickle

import torch

# Env
from networks import define_net
from data_loaders import *
from options import parse_args
from train_test import train, test
from collections import OrderedDict

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

data_cv_path = os.path.join(opt.dataroot, opt.split_directory)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
	print("*******************************************")
	load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
	model_ckpt = torch.load(load_path, map_location=device)

	#### Loading Env
	model_state_dict = model_ckpt['model_state_dict']
	if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

	model = define_net(opt, None)
	if isinstance(model, torch.nn.DataParallel): 
		print("True")
		model = model.module

	omic_model = 0

	new_state_dict = OrderedDict()
	for k, v in model_state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	print('Loading the model from %s' % load_path)
	model.load_state_dict(new_state_dict)


	### 3.2 Evalutes Train + Test Error, and Saves Model
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, vae_loss = test(opt, model, data, 'test', device, omic_model)

	if opt.task == 'surv':
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results.append(cindex_test)


	### 3.3 Saves Model
	pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_pred_test.pkl' % (opt.model_name)), 'wb'))


print('Split Results:', results)
print("Average:", np.array(results).mean())
pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))