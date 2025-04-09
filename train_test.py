import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import VitkanDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
#from models.vae_basic_model import VaeBasicModel
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters,collate_MIL
from torch.utils.tensorboard import SummaryWriter
#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os
torch.autograd.set_detect_anomaly(True)
                
def train(opt, data, device, k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    
    tb_logger = SummaryWriter(f'tensorboard/{opt.exp_name}/{k}') # log directory -mg

    custom_data_loader = VitkanDatasetLoader(opt, data, split='train', mode=opt.mode)
    
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True)
    
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    opt.omics_dims = custom_data_loader.omics_dims
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    # vae omic model is not supported at the moment -mg
    if opt.omic_model == 'vae':
        #omic_model = omic_vae(opt)
        print('INITIALIZING OMIC_VAE MODEL')
        omic_model = VaeBasicModel(opt)
        if opt.net_VAE == 'fc_sep':
            opt.add_channel = False
            opt.ch_separate = True
        elif opt.net_VAE == 'fc':
            opt.add_channel = False
            opt.ch_separate = False
        elif opt.net_VAE == 'conv_1d':
            opt.add_channel = True
            opt.ch_separate = False
    else: omic_model = 0

    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)
    old_cindex = 0
    patient_count = 0
    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):

        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()
        if opt.omic_model == 'vae': omic_model.set_train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch =  0

        for batch_idx, (x_path,  x_omic, censor, survtime, omics_dict) in enumerate(train_loader):


            if opt.omic_model == 'vae':
                omic_model.set_input(input_dict=omics_dict)
                vae_output = omic_model.forward()
            else: vae_output=0
            
            censor = censor.to(device) if "surv" in opt.task else censor

            _, pred = model(x_path=x_path.to(device),  x_omic=x_omic.to(device), vae_omic = vae_output, omic_model =opt.omic_model)

            #print('pred',pred.shape)

            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0

            #print('loss_cox',loss_cox)
        
            loss_reg = define_reg(opt, model)
            
            #print(opt.lambda_cox, loss_cox ,opt.lambda_nll, loss_nll ,opt.lambda_reg, loss_reg)
            
            #loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_regf
            loss = opt.lambda_cox*loss_cox
            #print('loss',loss) 
                
            loss_epoch += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.omic_model=='vae':
                omic_model.update()
              
            if opt.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   

            
            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx+1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch+1, opt.niter+opt.niter_decay, batch_idx+1, len(train_loader), loss.item()))

        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)  if opt.task == 'surv' else None
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)  if opt.task == 'surv' else None
    
            loss_test, cindex_test, pvalue_test, surv_acc_test, _, pred_test, vae_loss = test(opt, model, data, 'test', device, omic_model)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)


            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%d_pred_test.pkl' % (opt.model_name, k, epoch)), 'wb'))

            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
                    print('vae_loss:', vae_loss)
                    tb_logger.add_scalar('Train_loss', loss_epoch, epoch)
                    tb_logger.add_scalar('Test_loss', loss_test, epoch)
                    tb_logger.add_scalar('Train C-Index', cindex_epoch, epoch)
                    tb_logger.add_scalar('Test C-index', cindex_test, epoch)

            if opt.task == 'surv':
                print(old_cindex)
                print(patient_count)
                
                c = cindex_test - old_cindex
                if c > opt.patience:
                    patient_count=0
                    old_cindex = cindex_test
                    best_model_state = model.state_dict
                    print('Saved best model')
                    torch.save({'model_state_dict':best_model_state}, 'best_model.pth')
                else: 
                    patient_count+=1
                    if patient_count > opt.patient_limit:
                        print("Early stopping at Epoch %d" % epoch)
                        break
    tb_logger.close()
    return model, optimizer, metric_logger, omic_model


def test(opt, model, data, split, device, omic_model):
    model.eval()
    if opt.omic_model == 'vae': omic_model.set_eval()
    custom_data_loader = VitkanDatasetLoader(opt, data, split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False, collate_fn=mixed_collate)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all = None
    loss_test = 0
    for batch_idx, (x_path, x_omic, censor, survtime,  omics_dict) in enumerate(test_loader):

        if opt.omic_model == 'vae':
            omic_model.set_input(input_dict=omics_dict)
            vae_output, vae_loss = omic_model.test()
  
        else: vae_output, vae_loss = 0, 0
        censor = censor.to(device) if "surv" in opt.task else censor
        
        
        with torch.no_grad():
            _, pred = model(x_path=x_path.to(device), x_omic=x_omic.to(device), vae_omic = vae_output, omic_model =opt.omic_model)

            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, omic_model) if  opt.omic_model == 'vae' else define_reg(opt, model) 
            loss_nll =  0
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
            loss_test += loss.data.item()

            if opt.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   

    
    
    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_test =  None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, vae_loss
