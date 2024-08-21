import argparse
import os

import torch

### Parser


#TODO: modify model arguments
#DONE: data arguments 
#DONE: task arguments

def parse_args():

    #data arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/media/nfs/TCGA_SLIDES', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='test', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--split_directory', type=str, default='splits/VIT_CTRANS_mirna_cnv_lusc_1img.pkl', help='name of the split')
    parser.add_argument('--wandb', type=int, default= 1, help='initialize wandb parameter search')
    
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default=None, help='mode')
    parser.add_argument('--model_name', type=str, default=None, help='mode')
    parser.add_argument('--visnet', type=str, default=None, help='vit | swin | ctrans | histossl')
    parser.add_argument('--transform', type= str, default=None, help= 'standart_scaler, robust, normalize')

    parser.add_argument('--use_vgg_features', type=int, default=1, help='Use pretrained embeddings')
    parser.add_argument('--use_vit_features',type = int, default=1,help = "use wsi features from pre trained VIT model")
    parser.add_argument('--use_rnaseq', type=int, default=0, help='Use RNAseq data.')
    
    # task argumens 
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--useRNA', type=int, default=0) # Doesn't work at the moment...:(
    parser.add_argument('--useSN', type=int, default=0)
    parser.add_argument('--omic_act_type', type=str, default='ReLU', help='ELU, ReLU, Leaky_ReLU')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')

    parser.add_argument('--input_size_omic', type=int, default=9, help="input_size for omic vector") #get size of omic vector -mg
    parser.add_argument('--input_size_path', type=int, default=224, help="input_size for path images") #changed this to 224
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--save_at', type=int, default=20, help="adsfasdf")
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=0, type=int)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--finetune', default=1, type=int, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--reg_type', default='omic', type=str, help="regularization type")
    
    
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=0, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")

    parser.add_argument('--lambda_cox', type=float, default=1.5)
    parser.add_argument('--lambda_reg', type=float, default=0)
    parser.add_argument('--lambda_nll', type=float, default=1)


    #model arguments 

    parser.add_argument('--fusion_type', type=str, default=None, help='concat | pofusion | cross_attention | self_attention')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--use_bilinear', type=int, default=0)
    parser.add_argument('--path_gate', type=int, default=0)
    parser.add_argument('--grph_gate', type=int, default=0)
    parser.add_argument('--omic_gate', type=int, default=0)
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--grph_dim', type=int, default=32)
    parser.add_argument('--omic_dim', type=int, default=32)
    parser.add_argument('--path_scale', type=int, default=1)
    parser.add_argument('--grph_scale', type=int, default=1)
    parser.add_argument('--omic_scale', type=int, default=1)
    parser.add_argument('--mmhid', type=int, default=64)

    parser.add_argument('--init_type', type=str, default=None, help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_method', default=None, type=str, help='AlphaDropout, Dropout')
    parser.add_argument('--dropout_rate', default=None, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--use_edges', default=1, type=float, help='Using edge_attr')
    parser.add_argument('--pooling_ratio', default=None, type=float, help='pooling ratio for SAGPOOl')
    parser.add_argument('--lr', default=None, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--weight_decay', default=None, type=float, help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--GNN', default='GCN', type=str, help='GCN | GAT | SAG. graph conv mode for pooling')
    parser.add_argument('--patience', default=0.002, type=float)
    parser.add_argument('--patient_limit', default=10, type=int)
    parser.add_argument('--patient_count', default=0, type=int)
    parser.add_argument('--lr_decay_iters', default=0, type=int)
    
    parser.add_argument('--omic_model', type=str, default=None, 
                    help='omic model to be used, options: [vae | fcn | encoder | kan | fast_kan]')

    #kan model arguments
    parser.add_argument('--kan_hlayer', type=int, default=None)
    parser.add_argument('--kan_gridsize', type=int, default=None)
    
    # encoder model arguments
    parser.add_argument('--n_head', type=int, default=4) #also applies fcn model !!
    parser.add_argument('--n_layer', type=int, default=1)

    # vae model arguments, ignore parameters if omic model = fcn
    parser.add_argument('--net_VAE', type=str, default='fc',
                        help='specify the backbone of the VAE, default is the one dimensional CNN, options: [conv_1d | fc_sep | fc]')
    parser.add_argument('--omics_mode', type=str, default='c',
                            help='omics types would like to use in the model, options: [abc | ab | a | b | c]')
    parser.add_argument('--norm_type', type=str, default='batch',
                        help='the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]')
    parser.add_argument('--filter_num', type=int, default=8,
                        help='number of filters in the last convolution layer in the generator')
    parser.add_argument('--conv_k_size', type=int, default=9,
                        help='the kernel size of convolution layer, default kernel size is 9, the kernel is one dimensional.')
    parser.add_argument('--dropout_p', type=float, default=0.2,
                        help='probability of an element to be zeroed in a dropout layer, default is 0 which means no dropout.')
    parser.add_argument('--leaky_slope', type=float, default=0.2,
                        help='the negative slope of the Leaky ReLU activation function')
    parser.add_argument('--latent_space_dim', type=int, default=32,
                        help='the dimensionality of the latent space')
    parser.add_argument('--vae_init_type', type=str, default='normal',
                        help='choose the method of network initialization, options: [normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal]')
    parser.add_argument('--vae_init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal initialization methods')
    parser.add_argument('--vae_weight_decay', type=float, default=1e-4,
                    help='weight decay (L2 penalty)')
    parser.add_argument('--recon_loss', type=str, default='MSE',
                            help='chooses the reconstruction loss function, options: [BCE | MSE | L1]')
    parser.add_argument('--reduction', type=str, default='mean',
                            help='chooses the reduction to apply to the loss function, options: [sum | mean]')
    parser.add_argument('--k_kl', type=float, default=0.01,
                            help='weight for the kl loss')
    parser.add_argument('--k_embed', type=float, default=0.001,
                            help='weight for the embedding loss')

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
