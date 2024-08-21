import adabound
import torch
import torch.nn as nn

from torch.nn import init, Parameter

import torch.optim.lr_scheduler as lr_scheduler

from fusion import *
from options import parse_args
from utils import *

from efficient_kan import *

#from captum.attr import IntegratedGradients

################
# Network Utils
################
def define_net(opt, k):
    net = None
    act = define_act_layer(act_type=opt.act_type)
   
    if opt.mode == 'custom':
        net = CustomNet(opt=opt, act=act, k=k)
    elif opt.mode == 'SingleVisionNet':
        net = SingleVisionNet(opt=opt, act=act, k=k)
    elif opt.mode == 'DoubleFusionNet':
        net = DoubleFusionNet(opt=opt, act=act, k=k)
    elif opt.mode == 'SingleVisionNet_KAN':
        net = SingleVisionNet_KAN(opt=opt, act=act, k=k)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    elif opt.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.final_lr)
    elif opt.optimizer_type == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1)

    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(opt, model):
    loss_reg = None

    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    elif opt.reg_type == 'vae_omic':
        loss_reg = regularize_vae_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == 'Leaky_ReLU':
        act_layer = nn.LeakyReLU()
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


def define_trifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=3, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion_A':
        fusion = TrilinearFusion_A(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'pofusion_B':
        fusion = TrilinearFusion_B(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion

def define_attention(fusion_type, size = 32, head = 4):
    fusion = None
    if fusion_type == 'cross_attention':
       print("INITIALIZED CROSS ATTENTION FUSION")
       #fusion = CrossAttentionBlock(dim=size , hidden_dim= 64)
       fusion = cross(emb_dim = size, num_heads = head)
       mhid =32
    elif fusion_type == 'self_attention':
        print("INITIALIZED SELF ATTENTION FUSION")
        size = size*2
        #fusion = SelfAttentionFusion(feature_dim=size)
        fusion = selfattention(emb_dim= size, num_heads= head )
        mhid = 64 
    return fusion, mhid


############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, omic_act_type = None, dropout_method = None, dropout_rate=0.25, act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        if omic_dim == 3880:
            print('Initializing Large Omic Model') 
            hidden = [2048,1024,512,32]
        else:
            print('Initializing Small Omic Model') 
            hidden = [64, 48, 32, 32] 

        self.act = act
        
        self.dropout_method = dropout_method
        self.omic_act_type = omic_act_type

        # Define the dropout layer based on the specified method
        if self.dropout_method == 'AlphaDropout':
            self.dropout = nn.AlphaDropout
        elif self.dropout_method == 'Dropout':
            self.dropout = nn.Dropout
        else:
            raise ValueError("Unsupported dropout method:",dropout_method)

        if self.omic_act_type == 'ELU':
            self.omic_act_type = nn.ELU
        elif self.omic_act_type == 'ReLU':
            self.omic_act_type = nn.ReLU
        elif self.omic_act_type == 'Leaky_ReLU':
            self.omic_act_type = nn.LeakyReLU
        else:
            raise ValueError("Unsupported omic act type:", self.omic_act_type)
        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            self.omic_act_type(),
            self.dropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            self.omic_act_type(),
            self.dropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            self.omic_act_type(),
            self.dropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            self.omic_act_type(),
            self.dropout(p=dropout_rate, inplace=False))

        
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        x = torch.isfinite(x)
        features = self.encoder(x.float())
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        return features, out


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim = 9, n_head=4, n_layer = 2, omic_dim=32):
        super(TransformerEncoder, self).__init__()
       
        encoder_layer = nn.TransformerEncoderLayer(d_model = omic_dim, nhead= n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)
        self.projection = nn.Linear(input_dim, omic_dim)

    def forward(self, x):
        x = self.projection(x)
        features = self.encoder(x)
        return features
    
    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False
           
class CustomNet(nn.Module):
    def __init__(self, opt, act, k):
        super(CustomNet, self).__init__()
      
        if opt.omic_model == 'fcn':
            self.multihead_attn = nn.MultiheadAttention(32, 4)
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_method= opt.dropout_method,  dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)
        elif opt.omic_model == 'encoder':
            self.omic_net = TransformerEncoder(input_dim=opt.input_size_omic, omic_dim = opt.omic_dim, n_head = opt.n_head, n_layer=opt.n_layer)
        elif opt.omic_model == 'vae': pass
        self.fusion = define_trifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.grph_gate, gate3=opt.omic_gate, dim1=opt.path_dim, dim2=opt.grph_dim, dim3=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.grph_scale, scale_dim3=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False) #what is this ? -mg
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs['x_path'] #model takes path vector as parameter -mg
        grph_vec = kwargs['x_grph']

        if kwargs['omic_model'] == 'vae':      
            omic_vec = kwargs['vae_omic']
        elif kwargs['omic_model'] == 'fcn':     
            raw_omic, _ = self.omic_net(x_omic=kwargs['x_omic'])
            omic_vec, _ = self.multihead_attn(raw_omic, raw_omic, raw_omic)
        elif kwargs['omic_model'] == 'encoder':
            omic_vec = self.omic_net(x = kwargs['x_omic'])
        features = self.fusion(path_vec, grph_vec, omic_vec) 
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False
    
class SingleVisionNet(nn.Module):
    def __init__(self, opt, act, k):
        super(SingleVisionNet, self).__init__()

        print('INITIALIZING SINGLENET MODEL')
        mhid = 32
        if opt.omic_model == 'fcn':
            self.multihead_attn = nn.MultiheadAttention(32, opt.n_head)
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim,  omic_act_type = opt.omic_act_type, dropout_method= opt.dropout_method,  dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)
            
        elif opt.omic_model == 'encoder':
            self.omic_net = TransformerEncoder(input_dim=opt.input_size_omic, omic_dim = opt.omic_dim, n_head = opt.n_head, n_layer=opt.n_layer)
        elif opt.omic_model == 'vae': pass

        if opt.fusion_type == 'pofusion':
            mhid = 64
            self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.omic_gate, dim1=opt.path_dim, dim2=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, mmhid= mhid, dropout_rate=opt.dropout_rate)
        else:    
            self.fusion, mhid = define_attention(fusion_type=opt.fusion_type, size=32, head = opt.n_head)
        self.classifier = nn.Sequential(nn.Linear(mhid, opt.label_dim))
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False) #what is this ? -mg
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        self.visnet = opt.visnet

    def forward(self, **kwargs):
        
        if self.visnet == 'vit':
            path_vec = kwargs['x_path']
        else: path_vec = kwargs['x_grph']
           
        if kwargs['omic_model'] == 'vae':      
            omic_vec = kwargs['vae_omic']
        elif kwargs['omic_model'] == 'fcn':      
            raw_omic, _ = self.omic_net(x_omic=kwargs['x_omic'])
            omic_vec, _ = self.multihead_attn(raw_omic, raw_omic, raw_omic)
        elif kwargs['omic_model'] == 'encoder':
            omic_vec = self.omic_net(x = kwargs['x_omic'])

        features = self.fusion(path_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False
class SingleVisionNet_KAN(nn.Module):
    def __init__(self, opt, act, k):
        super(SingleVisionNet_KAN, self).__init__()

        print('INITIALIZING SINGLENET MODEL WITH KANS')
        mhid = 64
        if opt.omic_model == 'kan':
            self.omic_net = KAN(width=[9,opt.kan_hlayer,32], grid=opt.grid_zie, k=3, seed=0)
            self.fusion = KAN(width = [mhid,opt.kan_hlayer,mhid], grid = opt.grid_size)
        elif opt.omic_model == 'fast_kan':
            self.omic_net = KAN([opt.input_size_omic, opt.kan_hlayer, 32], grid_size = opt.kan_gridsize)
            self.fusion = KAN([mhid,opt.kan_hlayer,opt.label_dim], grid_size = opt.kan_gridsize)            
        self.visnet = opt.visnet    
    def forward(self, **kwargs):
        
        path_vec = kwargs['x_path']
        omic_vec, features = self.omic_net(kwargs['x_omic'])
        fused_vector = torch.cat([path_vec, omic_vec], dim = 1 )

        hazard, features = self.fusion(fused_vector)
        return  features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False    
    
class DoubleFusionNet(nn.Module):
    def __init__(self, opt, act, k):
        super(DoubleFusionNet, self).__init__()
        # Define Fusion Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=32,
                                                   nhead=2,
                                                   dim_feedforward=12)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_transform = nn.Linear(32, 32)

        print('INITIALIZING Double Fusion MODEL')
        if opt.omic_model == 'fcn':
            self.multihead_attn = nn.MultiheadAttention(32, opt.n_head)
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)
        elif opt.omic_model == 'encoder':
            self.omic_net = TransformerEncoder(input_dim=opt.input_size_omic, omic_dim = opt.omic_dim, n_head = opt.n_head, n_layer=opt.n_layer)
        elif opt.omic_model == 'vae': pass
        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.omic_gate, dim1=opt.path_dim, dim2=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False) #what is this ? -mg
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        

    def forward(self, **kwargs):

        path_vec = kwargs['x_path']
        grph_vec = kwargs['x_grph'] 
        src = torch.cat([path_vec.unsqueeze(0), grph_vec.unsqueeze(0)], dim=0)
        # Pass through the transformer
        output = self.transformer_encoder(src)
        aggregated_output = output.mean(dim=0)
        fused_vit = self.output_transform(aggregated_output)
               
        if kwargs['omic_model'] == 'vae':      
            omic_vec = kwargs['vae_omic']
        elif kwargs['omic_model'] == 'fcn':     
            raw_omic, _ = self.omic_net(x_omic=kwargs['x_omic'])
            omic_vec, _ = self.multihead_attn(raw_omic, raw_omic, raw_omic)
        elif kwargs['omic_model'] == 'encoder':
            omic_vec = self.omic_net(x = kwargs['x_omic'])
            
        features = self.fusion(fused_vit, omic_vec)
        
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False