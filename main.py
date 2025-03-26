import argparse
from GramGAN import GramGAN
from utils import *
from util import *
import torch
import numpy as np
import random
import torch.nn.init as init

def parse_args():
    desc = "Pytorch implementation of GramLIN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase',type=str,default='train',help='[train/test]')
    parser.add_argument('--dataset',type=str,default='newslthe',help='dataset_name')

    parser.add_argument('--iteration', type=int, default=400000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=300, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    
    parser.add_argument('--adv_weight', type=int, default=10, help='Weight for GAN')
    parser.add_argument('--lambda_GNN', type=float, default=10, help='weight for HDCE loss: HDCE(G(X), X)')
    parser.add_argument('--identity_weight', type=int, default=15, help='Weight for Identity')
    parser.add_argument('--aux_weight', type=int, default=10, help='Weight for Auxiliary classifiers')
    parser.add_argument('--alpha_entropy_weight',type=int,default=10,help='Weight for alpha_entropy1')
    parser.add_argument('--L1_weight',type=int,default=10,help='Weight for L1')
    parser.add_argument('--nce_layers', type=str, default='2,3,4', help='compute NCE loss on which layers')
    parser.add_argument('--patchnce_layers', type=str, default='0,1,2,4,5', help='compute PatchNCE loss on which layers')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=10, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--result_dir', type=str, default='../autodl-tmp/result45/', help='Directory name to save the results')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--resume', type=str2bool, default=True)
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
    parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
    parser.add_argument('--no_dropout', type=str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
    parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
    parser.add_argument('--lambda_NCE', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
    parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
    parser.add_argument('--flip_equivariance',
                            type=str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    
    parser.add_argument('--netF_nc', type=int, default=256)
 ##### Graph configs.
    parser.add_argument('--gnn_idt', action='store_true')
    parser.add_argument('--num_hop', type=int, default=2)
    parser.add_argument('--pooling_num', type=int, default=1)
    parser.add_argument('--down_scale', type=int, default=4)
    parser.add_argument('--pooling_ratio', type=str, default=[1,0.5], help='Ratio for pooling level | [ level0, level1, level2 ]')
    parser.add_argument('--nonzero_th', type=float, default=0.6)

    parser.set_defaults(pool_size=0)  # no image pooling    
    
    return check_args(parser.parse_args())
"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def weights_init(m):
    classname = m.__class__.__name__
    
      if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data) 
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)  
    
    
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)  
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0) 
    
       elif classname.find('InstanceNorm2d') != -1:
        if m.weight is not None:
            init.constant_(m.weight.data, 1.0)  
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)  


def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    set_seed(42) 
    # open session
    gan = GramGAN(args)
    
    gan.bulid_model()  
    gan.genA2B.apply(weights_init)
    gan.disGA.apply(weights_init)
    gan.disMA.apply(weights_init)
    gan.disLA.apply(weights_init)
    
    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
