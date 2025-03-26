import torch
import time
from torch.functional import norm
import torch.nn as nn
from torch.nn.init import ones_, xavier_normal_
from torch.nn.modules.activation import Sigmoid
from torch.nn.parameter import Parameter
import numpy as np
import os
from utils import gram_matrix
from torch.nn import init
import torch.nn.utils as utils
torch.set_printoptions(profile="full")


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = 0.07
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = torch.nn.functional.normalize(contrast_feature, p=2, dim=-1)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        
       
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
       
        logits = anchor_dot_contrast - logits_max.detach()
         

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        

        # compute log_prob
        explogit0=torch.exp(logits)
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
       

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling

        for i in range(8):
            setattr(self, 'Block' + str(i+1), ResnetGramGANBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock2 = nn.Sequential(*UpBlock2)


    def forward(self, input, label_target=None, device=None, layers=[0,1,2,3,4,5,6,7],layers2=[0,1,2,4] ,encode_only=False):
        if not encode_only:
            x0 = self.DownBlock(input)
            x1, S1, L1 = getattr(self, 'Block1')(x0, label_target, device)
            x2, S2, L2 = getattr(self, 'Block2')(x1, label_target, device)
            x3, S3, L3 = getattr(self, 'Block3')(x2, label_target, device)
            x4, S4, L4 = getattr(self, 'Block4')(x3, label_target, device)
            x5, S5, L5 = getattr(self, 'Block5')(x4, label_target, device)
            x6, S6, L6 = getattr(self, 'Block6')(x5, label_target, device)
            x7, S7, L7 = getattr(self, 'Block7')(x6, label_target, device)
            x8, S8, L8 = getattr(self, 'Block8')(x7, label_target, device)
            out = self.UpBlock2(x8)
            return out, S1, S2, S3, S4, S5, S6, S7, S8, L1, L2, L3, L4, L5, L6, L7, L8
        
        else:
            if -1 in layers:
                layers.append(len(self.DownBlock) + len(self.UpBlock2) + 8)  # Assuming 8 ResnetGramGANBlock

            feats = []
            feats2= []
            x0 = self.DownBlock(input)
            if 0 in layers:
                feats.append(x0)
            if 0 in layers2:
                feats2.append(x0)
            x = x0
            S_all, L_all = [], []
            for i in range(8):
            #print("i:",i)
                x, S, L = getattr(self, 'Block' + str(i+1))(x, label_target, device)
                S_all.append(S)
                L_all.append(L)
                if i+1 in layers:
                    feats.append(x)
                if i+1 in layers2:
                    feats2.append(x)
                    #print("feats.shape",feats.shape)
                if i + 1 == layers[-1] and encode_only:
                    return feats,feats2  
        # out = self.UpBlock2(x)
        # #print("encode_only:",encode_only)
        if encode_only:
            return feats,feats2
        #print("nihaoma")
        return out, S_all[0], S_all[1], S_all[2], S_all[3], S_all[4], S_all[5], S_all[6], S_all[7],L_all[0], L_all[1], L_all[2], L_all[3], L_all[4], L_all[5], L_all[6], L_all[7]

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if gpu_ids:
        assert(torch.cuda.is_available())
        device = torch.device(f'cuda:{gpu_ids[0]}')
        net.to(device)
    init_weights(net, init_type, init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[],NC=256):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
      
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=NC)
        
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=NC)
        #print(net)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)    

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x, dim=1):
        # norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        # out = x.div(norm + 1e-7)
        # FDL: To avoid sqrting 0s, which causes nans in grad
        norm = (x + 1e-7).pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # print("feats shape:",feats.shape)
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            #print("feat shape:",feat.shape)
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1) 
                
            else:
                x_sample = feat_reshape.flatten(0, 1)
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.reshape([B, H, W, x_sample.shape[-1]]).permute(0, 3, 1, 2)
            return_feats.append(x_sample)
        return return_feats, return_ids




class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGramGANBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetGramGANBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = GramLIN(dim)  
        self.relu1 = nn.ReLU(True)

    def forward(self, x, target_label,device):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, target_label,device)
        S=self.norm1.S_A
        L=self.norm1.gramA_L1
        out = self.relu1(out)
        return out + x,S,L

class GramLIN(nn.Module):
    def __init__(self, num_features,eps=1e-5):
        super(GramLIN, self).__init__()
        self.eps = eps
        #weight_1:W
        self.weight_1=Parameter(torch.Tensor(num_features*num_features,4))
        self.gamma_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))

        self.X = Parameter(torch.Tensor(1,4))
        self.num_features=num_features
        torch.nn.init.normal_(self.weight_1, mean=0, std=1)
        self.rho.data.fill_(0.9)
        self.gamma_1.data.fill_(1.0)
        self.beta_1.data.fill_(1.0)
        self.gamma_2.data.fill_(0.0)
        self.beta_2.data.fill_(0.0)
        self.X.data.fill_(1.0)

    def forward(self,input,target_label,device):
      
        input_device = input.device
        self.weight_1 = self.weight_1.to(input_device)
        self.gamma_1 = self.gamma_1.to(input_device)
        self.gamma_2 = self.gamma_2.to(input_device)
        self.beta_1 = self.beta_1.to(input_device)
        self.beta_2 = self.beta_2.to(input_device)
        self.rho = self.rho.to(input_device)
        newlabel = torch.zeros((input.shape[0], 1, 4), device=input_device)
        for i in range(4):
            if i == target_label:
                newlabel[:, :, i] = 1
            else:
                newlabel[:, :, i] = 0.05

        gram = gram_matrix(input, input_device).to(input_device)
        gram = gram.reshape(1, gram.shape[1] * gram.shape[2])
        theta = torch.mm(gram, self.weight_1)

        s_weight_1 = torch.sigmoid(self.weight_1)
        gramW = torch.mm(s_weight_1.t(), s_weight_1)
        gramW1 = torch.reciprocal(gramW)
        gramW2 = torch.sqrt(gramW1)
        gramA_ = torch.mm(torch.diag(torch.diag(gramW2)), gramW)
        gramA__ = torch.mm(gramA_, torch.diag(torch.diag(gramW2)))
        gramA = torch.div(gramA__, 4.0)
        self.gramA_L1 = torch.norm(gramA, p=1) - torch.trace(gramA)

        self.S_A = -1.0 / (torch.log2(torch.trace(torch.mm(gramA, gramA))))

        newweight = torch.mm(theta, newlabel[0].t())
        for i in range(1, input.shape[0]):
            newweight = torch.cat((newweight, torch.mm(theta, newlabel[i].t())), 0)
        # for i in range(input.shape[0]):
        #     if i==0:
        #         newweight=torch.mm(theta,target_label[i].t())
        #     else:newweight=torch.cat((newweight,torch.mm(target_label[i],self.weight[0])),0)
        self.gamma = self.gamma_1 * newweight + self.gamma_2
        beta = self.beta_1 * newweight + self.beta_2

        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + beta.expand(input.shape[0], -1, -1, -1)
        return out

# class GramLIN(nn.Module):
#     def __init__(self, num_features, eps=1e-5):
#         super(GramLIN, self).__init__()
#         self.eps = eps
#         self.weight_1 = Parameter(torch.Tensor(num_features * num_features, 4))
#         self.gamma_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
#         self.gamma_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
#         self.beta_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
#         self.beta_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
#         self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
#         self.num_features = num_features
#         torch.nn.init.normal_(self.weight_1, mean=0, std=1)
#         self.rho.data.fill_(0.9)
#         self.gamma_1.data.fill_(1.0)
#         self.beta_1.data.fill_(1.0)
#         self.gamma_2.data.fill_(0.0)
#         self.beta_2.data.fill_(0.0)

#     def forward(self, input, target_label, device):
#         #if target_label is None:
#             #raise ValueError("target_label is None. It must be a valid tensor.")

#         input_device = input.device
#         self.weight_1 = self.weight_1.to(input_device)
#         self.gamma_1 = self.gamma_1.to(input_device)
#         self.gamma_2 = self.gamma_2.to(input_device)
#         self.beta_1 = self.beta_1.to(input_device)
#         self.beta_2 = self.beta_2.to(input_device)
#         self.rho = self.rho.to(input_device)
#         #target_label = target_label.view(-1, 4).t()
#         newlabel = torch.zeros((input.shape[0], 4), device=input_device)  
#         for i in range(4):
#             if i == target_label:
#                 newlabel[:, i] = 1
#             else:
#                 newlabel[:, i] = 0.05

#         gram = gram_matrix(input, device).to(input_device)
#         gram = gram.reshape(1, gram.shape[1] * gram.shape[2])
#         theta = torch.mm(gram, self.weight_1).to(input_device) 

#         s_weight_1 = torch.sigmoid(self.weight_1)
#         gramW = torch.mm(s_weight_1.t(), s_weight_1)
#         gramW1 = torch.reciprocal(gramW)
#         gramW2 = torch.sqrt(gramW1)
#         gramA_ = torch.mm(torch.diag(torch.diag(gramW2)), gramW)
#         gramA__ = torch.mm(gramA_, torch.diag(torch.diag(gramW2)))
#         gramA = torch.div(gramA__, 4.0)
#         self.gramA_L1 = torch.norm(gramA, p=1) - torch.trace(gramA)

#         self.S_A = -1.0 / (torch.log2(torch.trace(torch.mm(gramA, gramA))))

#         newweight = torch.mm(theta, newlabel.t()).to(input_device)  
#         for i in range(1, input.shape[0]):
#             newweight = torch.cat((newweight, torch.mm(theta, newlabel[i].t())), 0)

#         self.gamma = self.gamma_1 * newweight + self.gamma_2
#         beta = self.beta_1 * newweight + self.beta_2

#         in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
#         out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
#         ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
#         out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
#         out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
#         out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + beta.expand(input.shape[0], -1, -1, -1)
#         return out

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=3, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.head1 = nn.Sequential(
                nn.Linear(ndf *mult, ndf *mult),
                nn.ReLU(inplace=True),
                nn.Linear(ndf *mult, 128)
            )
        self.head2 = nn.Linear(ndf *mult, 8)

        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        
        self.lamda = nn.Parameter(torch.zeros(1))
        self.total_time=0
        self.gap_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 4, bias=False)) #4分类
        self.gmp_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 4, bias=False))
        self.lina=nn.Linear(4, 4,bias=False)
        self.linm=nn.Linear(4, 4,bias=False)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, True)
        self.pad_2 = nn.ReflectionPad2d(1)

        self.lamda_2 = nn.Parameter(torch.zeros(1))
        self.conv1x121 = nn.Conv2d(ndf * mult,ndf * mult, kernel_size=1)
        self.leaky_relu221=nn.LeakyReLU(0.2, True)
        self.conv1x122 = nn.Conv2d(ndf * mult, ndf * mult, kernel_size=1)
        self.leaky_relu222=nn.LeakyReLU(0.2, True)
        for i in range(4):
            setattr(self, 'conv1*1_2_' + str(i), nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True))
        for i in range(4):
            setattr(self, 'conv_2_' + str(i), nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))

        self.model = nn.Sequential(*model)
    def forward(self, input,device):
        
        x0 = self.model(input)
        contra = torch.nn.functional.adaptive_avg_pool2d(x0, 8)
        xcon=contra.view(-1,contra.shape[1])
        
        epoch_start_time = time.time()
        heatmap_0 = torch.sum(x0, dim=1, keepdim=True)
        
        contra=self.head2(xcon)
        
        contra = contra.view(x0.shape[0], 1, -1)
        # print(f"contrashape:{contra.shape}")
        #real or fake
        x=x0
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        
        x = self.lamda*x + x0

        heatmap_1 = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)
        epoch_end_time = time.time()  
        epoch_time = epoch_end_time - epoch_start_time  
        self.total_time += epoch_time                                            
        print(f"disc_time:{self.total_time}")
        
        out_2=torch.zeros((input.shape[0],4),dtype=float)
        x=x0
        #classification
        gap_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit_2 = self.gap_fc_2(gap_2.view(x.shape[0], -1))

        gmp_2 = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit_2 = self.gmp_fc_2(gmp_2.view(x.shape[0], -1))
        gap_logit_2 = self.lina(gap_logit_2).unsqueeze(2)  # [batch_size, 4, 1]
        gmp_logit_2 = self.linm(gmp_logit_2).unsqueeze(2)
 
        cam_logit_2_0 = torch.cat([gap_logit_2, gmp_logit_2], 2)
        cam_logit_2 = torch.mean(cam_logit_2_0,dim=2,keepdim=False) 
        Softmax = nn.Softmax(dim=1)
        cam_logit_2=Softmax(cam_logit_2)

        x = self.lamda_2*x + x0

        heatmap_2=torch.Tensor(x.shape[0],0,x.shape[2],x.shape[3]).to(device)
        
        for i in range(4):  
            x = x0

            gap_weight_2 = list(self.gap_fc_2.parameters())[0][i].unsqueeze(0) #[1, ndf * mult]
            gap_2 = x * gap_weight_2.unsqueeze(2).unsqueeze(3) #[1, channels, 1, 1]

            gmp_weight_2 = list(self.gmp_fc_2.parameters())[0][i].unsqueeze(0)
            gmp_2 = x * gmp_weight_2.unsqueeze(2).unsqueeze(3)
            gap_2 = self.conv1x121(gap_2)  # [batch_size, channels, H, W]
            gap_2 =self.leaky_relu221(gap_2)
            gmp_2 = self.conv1x122(gmp_2)  # [batch_size, channels, H, W]
            gmp_2 =self.leaky_relu222(gmp_2)

            x = torch.cat([gap_2, gmp_2], 1)
            x = self.leaky_relu_2(getattr(self, 'conv1*1_2_' + str(i))(x))

            heatmap_2_0 = torch.sum(x, dim=1, keepdim=True)
            heatmap_2=torch.cat((heatmap_2,heatmap_2_0),1)

            x = self.pad_2(x)
            out_2_0 = getattr(self, 'conv_2_' + str(i))(x)
            if i==0:
                out_2 = torch.mean(out_2_0,axis=(2,3),keepdim=False) 
            else:out_2=torch.cat((out_2,torch.mean(out_2_0,axis=(2,3),keepdim=False)),1)

        Softmax = nn.Softmax(dim=1)
        out_2 = Softmax(out_2)

        return contra,out, cam_logit,out_2,cam_logit_2,heatmap_0,heatmap_1,heatmap_2[:,0],heatmap_2[:,1],heatmap_2[:,2],heatmap_2[:,3] 
#     def forward(self, input,device):
       
#         x0 = self.model(input)
#         epoch_start_time = time.time()
#         heatmap_0 = torch.sum(x0, dim=1, keepdim=True)

#         #real or fake
#         x=x0
#         gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
#         gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
#         gap_weight = list(self.gap_fc.parameters())[0]
#         gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

#         gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
#         gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
#         gmp_weight = list(self.gmp_fc.parameters())[0]
#         gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

#         cam_logit = torch.cat([gap_logit, gmp_logit], 1)
#        
#         x = torch.cat([gap, gmp], 1)
#         x = self.leaky_relu(self.conv1x1(x))
        
#         x = self.lamda*x + x0

#         heatmap_1 = torch.sum(x, dim=1, keepdim=True)

#         x = self.pad(x)
#         out = self.conv(x)
#         epoch_end_time = time.time()  
#         epoch_time = epoch_end_time - epoch_start_time  
#         self.total_time += epoch_time     
#         print(f"disc_time:{self.total_time}")
#         
#         out_2=torch.zeros((input.shape[0],4),dtype=float)
#         x=x0
#         #classification
#         gap_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
#         gap_logit_2 = self.gap_fc_2(gap_2.view(x.shape[0], -1))

#         gmp_2 = torch.nn.functional.adaptive_max_pool2d(x, 1)
#         gmp_logit_2 = self.gmp_fc_2(gmp_2.view(x.shape[0], -1))
        
       


#         gap_logit_2 = self.lina(gap_logit_2).unsqueeze(2)  # [batch_size, 4, 1]
#         gmp_logit_2 = self.linm(gmp_logit_2).unsqueeze(2) 
#         cam_logit_2_0 = torch.cat([gap_logit_2, gmp_logit_2], 2)
#         cam_logit_2 = torch.mean(cam_logit_2_0,dim=2,keepdim=False) 
#         Softmax = nn.Softmax(dim=1)
#         cam_logit_2=Softmax(cam_logit_2)

#         x = self.lamda_2*x + x0

#         heatmap_2=torch.Tensor(x.shape[0],0,x.shape[2],x.shape[3]).to(device)
        
#         for i in range(4): 
#             x = x0

#             gap_weight_2 = list(self.gap_fc_2.parameters())[0][i].unsqueeze(0) #[1, ndf * mult]
#             gap_2 = x * gap_weight_2.unsqueeze(2).unsqueeze(3) #[1, channels, 1, 1]

#             gmp_weight_2 = list(self.gmp_fc_2.parameters())[0][i].unsqueeze(0)
#             gmp_2 = x * gmp_weight_2.unsqueeze(2).unsqueeze(3)
    
            
#             gap_2 = self.conv1x121(gap_2)  # [batch_size, channels, H, W]
#             gmp_2 = self.conv1x122(gmp_2)  # [batch_size, channels, H, W]

            
#             gap_2 =self.leaky_relu221(gap_2)
#             gmp_2 =self.leaky_relu222(gmp_2)
#             x = torch.cat([gap_2, gmp_2], 1)
#             x = self.leaky_relu_2(getattr(self, 'conv1*1_2_' + str(i))(x))

#             heatmap_2_0 = torch.sum(x, dim=1, keepdim=True)
#             heatmap_2=torch.cat((heatmap_2,heatmap_2_0),1)

#             x = self.pad_2(x)
#             out_2_0 = getattr(self, 'conv_2_' + str(i))(x)
#             if i==0:
#                 out_2 = torch.mean(out_2_0,axis=(2,3),keepdim=False) 
#             else:out_2=torch.cat((out_2,torch.mean(out_2_0,axis=(2,3),keepdim=False)),1)

#         Softmax = nn.Softmax(dim=1)
#         out_2 = Softmax(out_2)
                                                         
            
        
#         return out, cam_logit,out_2,cam_logit_2,heatmap_0,heatmap_1,heatmap_2[:,0],heatmap_2[:,1],heatmap_2[:,2],heatmap_2[:,3]
    


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
