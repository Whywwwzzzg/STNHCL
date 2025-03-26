from __future__ import print_function
from packaging import version
import time
import torch
from torch import nn
import math
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch.factory import KNNGraph
from dgl.nn.pytorch import TAGConv, GATConv, GATv2Conv
import torch.nn.functional as F
import dgl.backend as B
import dgl.function as fn
import dgl
import numpy as np
from torch.nn import init
from torch_nn import BasicConv, batched_index_select, act_layer

eps = 1e-7

############
# utils
############

def initialize_memberships(batch_size, n_points, n_clusters, device):
    """
    Initialize the membership matrix for Fuzzy C-Means clustering.

    Args:
        batch_size: int
        n_points: int
        n_clusters: int
        device: torch.device

    Returns:
        memberships: tensor (batch_size, n_points, n_clusters)
    """
    # Randomly initialize the membership matrix ensuring that the sum over clusters for each point is 1
    memberships = torch.rand(batch_size, n_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)
    return memberships

def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
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
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=''):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     device = torch.device(f'cuda:{gpu_ids[0]}')  # 将字符串转换为设备对象
    #     net.to(device)
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

############
# Functions
############

def cos_distance_softmax(x):
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)

def soft_k_means(x, n_clusters, epsilon=5e-2, max_iter=100, T=0.15):
    """
    Soft k-means clustering

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations
        T: float, temperature parameter controlling the "softness" of the assignment

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_dims, num_points, _ = x.size()
    
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)
    
    print(f"x:{x.max()}")
    # Initialize the cluster centers randomly from the data points
    centers = x[:, torch.randperm(num_points)[:n_clusters], :].transpose(1, 2).contiguous()
    centers = centers.permute(0, 2, 1)
    
    
    for iteration in range(max_iter):
        
        # Calculate distances between points and cluster centers
        dist = torch.cdist(x, centers, p=2) 
        # diff = x.unsqueeze(2) - centers.unsqueeze(1)  # Shape: (batch_size, num_points, n_clusters, num_dims)
        # dist = torch.norm(diff, p=2, dim=-1)  # Shape: (batch_size, num_points, n_clusters)

        # Soft assignment with temperature parameter T
        memberships = F.softmax(-dist / 0.2, dim=-1)  # Shape: (batch_size, num_points, n_clusters)
        
        
 
        # Update cluster centers
        weights = memberships.permute(0, 2, 1)  # Shape: (batch_size, n_clusters, num_points)
        numerator = torch.bmm(weights, x)  # Shape: (batch_size, n_clusters, num_dims)
        denominator = weights.sum(dim=-1, keepdim=True)  # Shape: (batch_size, n_clusters, 1)
        new_centers = numerator / denominator
       

        # Check convergence
        if torch.norm(new_centers - centers) < epsilon:
            print(f"iteration:{iteration}")
            break

        centers = new_centers

    return memberships, centers
def fuzzy_c_means(x, n_clusters, m=2, epsilon=1e-6, max_iter=1000):
    """
    Fuzzy C-Means clustering

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        m: float, fuzziness parameter
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_dims, num_points, _ = x.size()
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)
    
    start_time = time.time()

    # Initialize the membership matrix
    memberships = initialize_memberships(batch_size, num_points, n_clusters, x.device)
    

    # Initialize cluster centers
    centers = torch.zeros(batch_size, num_dims, n_clusters, device=x.device)
    prev_memberships = torch.zeros_like(memberships)
    
    prev_memberships = prev_memberships.to(device)
    start_time = time.time()
    for iteration in range(max_iter):
        # Update cluster centers using vectorized operations
        weights = memberships ** m
        denominator = weights.sum(dim=1, keepdim=True)
        numerator = torch.bmm(weights.permute(0, 2, 1), x)
        
        centers = numerator / denominator.permute(0, 2, 1)
        
    
        # Update memberships using vectorized operations
        diff = x.unsqueeze(2) - centers.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=3)  # Euclidean distance
        memberships = 1.0 / (dist ** (2 / (m - 1)))

        # Normalize memberships
        memberships_sum = memberships.sum(dim=2, keepdim=True)
        memberships /= memberships_sum

        # Check convergence
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
    
        prev_memberships = memberships

    end_time = time.time()
    epoch_time = end_time - start_time

#     print(f"prev_membershipsdevice:{prev_memberships.device}")
#     start_time = time.time()
#     for iteration in range(max_iter):
#         # Update cluster centers
#         for cluster in range(n_clusters):
#             # Calculate the denominator
#             weights = memberships[:, :, cluster] ** m
#             print(f"weightsdevice:{weights.device}")
#             denominator = weights.sum(dim=1, keepdim=True)
#             # Update centers
#             numerator = (weights.unsqueeze(2) * x).sum(dim=1)
#             centers[:, :, cluster] = numerator / denominator

#         # Update memberships
#         for cluster in range(n_clusters):
#             diff = x - centers[:, :, cluster].unsqueeze(1)
#             dist = torch.norm(diff, p=2, dim=2)  # Euclidean distance
#             memberships[:, :, cluster] = 1.0 / (dist ** (2 / (m - 1)))

#         # Normalize the memberships such that each point's memberships across clusters sum to 1
#         memberships_sum = memberships.sum(dim=2, keepdim=True)
#         memberships = memberships / memberships_sum

#         # Check convergence: stop if memberships do not change significantly
#         if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
#             break
#         prev_memberships = memberships.clone()
        
#     end_time = time.time()  
#     epoch_time = end_time - start_time  
                                                      
            
    print(f"iteration_time:{epoch_time}")
    return memberships, centers


def construct_hyperedges(x, num_clusters, threshold=0.15, m=2):
    """
    Constructs hyperedges based on fuzzy c-means clustering.

    Args:
        x (torch.Tensor): Input point cloud data with shape (batch_size, num_dims, num_points, 1).
        num_clusters (int): Number of clusters (hyperedges).
        threshold (float): Threshold value for memberships to consider a point belonging to a cluster.
        m (float): Fuzzifier for fuzzy c-means clustering.

    Returns:
        hyperedge_matrix (torch.Tensor): Tensor of shape (batch_size, n_clusters, num_points_index).
            Represents each cluster's points. Padded with -1 for unequal cluster sizes.
        point_hyperedge_index (torch.Tensor): Tensor of shape (batch_size, num_points, cluster_index).
            Indicates the clusters each point belongs to. Padded with -1 for points belonging to different numbers of clusters.
        hyperedge_features (torch.Tensor): Tensor of shape (batch_size, num_dims, n_clusters).
            The center of each cluster, serving as the feature for each hyperedge.
    """
    
    with torch.no_grad():
        x = x.detach()  # Detach x from the computation graph
        device = x.device
        batch_size, num_dims, num_points, _ = x.shape
        # print(f"x:{x.max()}")
        # Get memberships and centers using the fuzzy c-means clustering
        memberships, centers = soft_k_means(x, 9, T=0.2)
        # print(f"memberships:{memberships.max()}")
        # Create hyperedge matrix to represent each hyperedge's points
        # Initialized with -1s for padding
        hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_points, dtype=torch.long, device=device)
#         for b in range(batch_size):
#             for c in range(num_clusters):
                
#                 idxs = torch.where(memberships[b, :, c] > threshold)[0]
#                 hyperedge_matrix[b, c, :len(idxs)] = idxs
        
        
        
        # Create point to hyperedge index to indicate which hyperedges each point belongs to
        # Initialized with -1s for padding
        max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
        point_hyperedge_index = -torch.ones(batch_size, num_points, max_edges_per_point, dtype=torch.long, device=device)
        for c in range(num_clusters):
            # print(f"memberships[:, :, c]:{memberships[:, :, c].max()}")
            # print(f"memberships[:, :, c]min:{memberships[:, :, c].min()}")
            mask = memberships[:, :, c] > threshold
            for b in range(batch_size):
                idxs = torch.where(mask[b])[0]
                hyperedge_matrix[b, c, :len(idxs)] = idxs
        for b in range(batch_size):
            for p in range(num_points):
                idxs = torch.where(memberships[b, p, :] > threshold)[0]
                point_hyperedge_index[b, p, :len(idxs)] = idxs
    # Create hyperedge matrix to represent each hyperedge's points
        # Initialized with -1s for padding
        # hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_points, dtype=torch.long, device=device)
        
        # Create point to hyperedge index to indicate which hyperedges each point belongs to
        # Initialized with -1s for padding
#         max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
#         point_hyperedge_index = -torch.ones(batch_size, num_points, max_edges_per_point, dtype=torch.long, device=device)

#         # Vectorized operation to fill hyperedge_matrix
#         mask = memberships > threshold
#         hyperedge_indices = torch.arange(num_points, device=device).expand(batch_size, num_points)
#         for c in range(num_clusters):
#             selected_points = mask[:, :, c]  # Shape: (batch_size, num_points)
#             num_selected_points = selected_points.sum(dim=-1)  # Shape: (batch_size,)
#             point_indices = hyperedge_indices[selected_points].view(batch_size, -1)
#             # Fill hyperedge_matrix with selected point indices
#             hyperedge_matrix[:, c, :num_selected_points.max()] = point_indices[:, :num_selected_points.max()]

#         # Vectorized operation to fill point_hyperedge_index
#         hyperedge_assignments = torch.where(mask, torch.arange(num_clusters, device=device).view(1, 1, -1), torch.tensor(-1, device=device))
#         sorted_hyperedge_assignments, _ = torch.sort(hyperedge_assignments, dim=-1, descending=True)
#         point_hyperedge_index = sorted_hyperedge_assignments[:, :, :max_edges_per_point]
#         print(f"nihao")
    # Return the three constructed tensors
    return hyperedge_matrix, point_hyperedge_index, centers

def nonzero_graph(x, th, exist_adj=None):
    
    if exist_adj is None:
        if B.ndim(x) == 2:
            x = B.unsqueeze(x, 0)
        n_samples, n_points, _ = B.shape(x)

        dist = torch.bmm(x, x.transpose(2,1)).squeeze().detach()
        base = torch.zeros_like(dist).cuda()
        base[dist>th] = 1
        adj = sparse.csr_matrix(B.asnumpy((base).squeeze(0)))
        
        #print(f'adj:{adj}')
    else:
        #print("nihao")
        if isinstance(exist_adj, torch.Tensor):
            
            adj = sparse.csr_matrix(B.asnumpy((exist_adj)))
        else:
            adj = exist_adj
        
        
    
    g = None
    
    if isinstance(adj, sparse.spmatrix):
        
        g = dgl.from_scipy(adj)
    
    elif isinstance(adj, (tuple, list)):
       
        g = dgl.graph(adj)
    elif isinstance(adj, torch.Tensor):
        g = dgl.graph(adj)
    
    if g is None:
        # print(f'Type of g: {type(g)}')
        # print(f'Type of adj: {type(adj)}')
        raise TypeError("Unsupported graph data type for adj.")

    #print("nihenhao")
    return g, adj
    return g, adj

class HypergraphConv2d(nn.Module):
    """
    Hypergraph Convolution based on the GIN mechanism
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(HypergraphConv2d, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Node to hyperedge transformation
        self.nn_node_to_hyperedge = BasicConv([in_channels, in_channels], act, norm, bias) # in_channels = 128, out_channels = 256
        self.nn_node_to_hyperedge = self.nn_node_to_hyperedge.to(self.device)
        # Hyperedge to node transformation
        self.nn_hyperedge_to_node = BasicConv([in_channels, out_channels], act, norm, bias)
        self.nn_hyperedge_to_node = self.nn_hyperedge_to_node.to(self.device)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init])).to(self.device)

    def forward(self, x, hyperedge_matrix, point_hyperedge_index, centers):
        # print(f"centers.shape: {centers.shape}")
        with torch.no_grad():
            
            # Check and append dummy node to x if not present
            if not torch.equal(x[:, :, -1, :], torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=self.device)): 
                dummy_node = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=self.device)
                x = torch.cat([x, dummy_node], dim=2) # (1, 128, 3137, 1)
            
            # Check and append dummy hyperedge to centers if not present
            if not torch.equal(centers[:, :, -1], torch.zeros((centers.size(0), centers.size(1), 1), device=self.device)):
                dummy_hyperedge = torch.zeros((centers.size(0), centers.size(1), 1), device=self.device)
                centers = torch.cat([centers, dummy_hyperedge], dim=2) # centers: (1, 128, 51)
        
        # Step 1: Aggregate node features to get hyperedge features
        node_features_for_hyperedges = batched_index_select(x, hyperedge_matrix)
        aggregated_hyperedge_features = node_features_for_hyperedges.sum(dim=-1, keepdim=True)
        
        # print(f"aggregated_hyperedge_features:{aggregated_hyperedge_features.max()}")
        aggregated_hyperedge_features = self.nn_node_to_hyperedge(aggregated_hyperedge_features) # (1, 128, 50)
        # print(f"aggregated_hyperedge_features:{aggregated_hyperedge_features.max()}")
        # Adding the hyperedge center features to the aggregated hyperedge features
        # print(f"aggregated_hyperedge_features shape: {aggregated_hyperedge_features.shape}")
        aggregated_hyperedge_features = aggregated_hyperedge_features.to(self.device)
        centers = centers.to(self.device)
        centers_squeezed = centers.unsqueeze(-1)  
        # print(f"centers_squeezed shape: {centers_squeezed.shape}")
#         aggregated_hyperedge_features = aggregated_hyperedge_features+(1 + self.eps) * centers_squeezed[:, :, :-1]
        
        centers_squeezed_permuted = centers_squeezed.permute(0, 2, 1, 3)  
        # print(f"aggregated_hyperedge_features.shape: {aggregated_hyperedge_features.shape}")
        # print(f"centers_squeezed_permuted.shape: {centers_squeezed_permuted.shape}")
        
        # print(f"aggregated_hyperedge_features shape: {aggregated_hyperedge_features.shape}")
        # print(f"centers_squeezed_permuted[:, :-1, :, :] shape: {centers_squeezed_permuted[:, :-1, :, :].shape}")

        aggregated_hyperedge_features = aggregated_hyperedge_features + (1 + self.eps) * centers_squeezed_permuted[:, :-1, :,:]
        
        # print(f"aggregated_hyperedge_features:{aggregated_hyperedge_features.max()}")
        # Step 2: Aggregate hyperedge features to update node features
        hyperedge_features_for_nodes = batched_index_select(aggregated_hyperedge_features.unsqueeze(-1), point_hyperedge_index)
        aggregated_node_features_from_hyperedges = self.nn_hyperedge_to_node(hyperedge_features_for_nodes.sum(dim=-1, keepdim=True))
        # print(f"aggregated_node_features_from_hyperedges:{aggregated_node_features_from_hyperedges.max()}")

        # Update original node features
        out = aggregated_node_features_from_hyperedges

        return out
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        bsz = x.shape[0]
        
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def top_k_graph(scores, g, h, k):
    '''
    :param scores: Node score
    :param g: teacher adjacent matrix
    :param h: Node feature
    :param k: Number of pooled node
    :return: pooled Adj, p
    '''
    device = scores.device  
    
   
    
    ## get high scored (score, node) = (values, new_h)
    values, idx = torch.topk(scores, max(2, int(k)))
    new_h = h[idx, :]  ## (k, dim)
    values = torch.unsqueeze(values, -1)  ## (k, score)
    new_h = torch.mul(new_h, values)  ## (k, dim)
    idx = idx.to(device)  

    ## get pooled adjacent matrix g
    ## increase connectiviy by hop2 as in original papaer.
    un_g = g.bool().float().to(device)  
    
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]  ## select idx rows
    un_g = un_g[:, idx]  ## select idx column
    g = norm_g(un_g)  ## random work by 1-hot graph
    g_numpy = B.asnumpy(g)  # Convert to numpy array and ensure it's on CPU
    g_sparse = sparse.csr_matrix(g_numpy)  # Convert to CSR sparse matrix

    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


###########
# Loss
###########
class GNNLoss(nn.Module):
    def __init__(self, opt, use_mlp=True):
        super(GNNLoss, self).__init__()
        self.l2norm = Normalize(2)
        self.oversample_ratio = 4 
        self.random_ratio = 0.25
        ## graph arguments
        self.num_hop = opt.num_hop
        self.pooling_num = 0
        self.down_scale = opt.down_scale
        self.pooling_ratio = opt.pooling_ratio
        self.nonzero_th = opt.nonzero_th
        self.total_time = 0
        self.gpu_ids = opt.gpu_ids
        self.nc = opt.netF_nc
        self.num_patch = opt.num_patches
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.i = 0
        self.use_mlp = use_mlp
        self.mlp_init = False
        self.k=9
        self.nett=HypergraphConv2d(256, 256, act=None, norm=None, bias=True)
        self.nets=HypergraphConv2d(256,256, act=None, norm=None, bias=True)

        self.criterion = NCESoftmaxLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            embed = Embed(input_nc, self.nc)
            pools = nn.ModuleList()
            gnn_pools = nn.ModuleList()
            if self.pooling_num>0:
                for i in range(self.pooling_num):
                    pools.append(Pool(self.nc))
                    gnn_pools.append(Encoder(self.nc, self.nc,1))
            gnn = Encoder(self.nc, self.nc, self.num_hop)
            print(f'self.gpu_ids:{self.gpu_ids}')
            if len(self.gpu_ids) > 0:
                gnn.cuda()
                pools.cuda()
                gnn_pools.cuda()
                embed.cuda()
                self.nett.cuda()
                self.nets.cuda()
                
            setattr(self, 'gnn_%d' % mlp_id, gnn)
            setattr(self, 'embed_%d' % mlp_id, embed)
            setattr(self, 'pools_%d' % mlp_id, pools)
            setattr(self, 'gnn_pools_%d' % mlp_id, gnn_pools)

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True
        
        
    def calc_hgnn(self,f_es_embed, f_et_embed, num_patches,  adj_s=None, adj_t=None):
        batch_dim_for_bmm = 1
        T = 0.07
        device = f_es_embed.device
        
            N, C = f_et_embed.shape
        # print(C)
        f_et_embed = f_et_embed.reshape(1, C, -1, 1).contiguous().to(device)
        f_es_embed = f_es_embed.reshape(1, C, -1, 1).contiguous().to(device)
        # print(f"f_et_embed_shape:{f_et_embed.shape}")
        # print(f"f_es_embed_shape:{f_es_embed.shape}")
        
        
        hyperedge_matrixt, point_hyperedge_indext, centerst = construct_hyperedges(f_et_embed, num_clusters=self.k)
        hyperedge_matrixt=hyperedge_matrixt.to(device)
        point_hyperedge_indext=point_hyperedge_indext.to(device)
        centerst=centerst.to(device)
        # hyperedge_matrixs,point_hyperedge_indexs, centerss= construct_hyperedges(f_es_embed.detach(), num_clusters=self.k)
        # hyperedge_matrixs=hyperedge_matrixs.to(device)
        # point_hyperedge_indexs=point_hyperedge_indexs.to(device)
        # centerss=centerss.to(device)
       
        # print(f"centerst:{centerst.max()}")
        f_gt=self.nett(f_et_embed,hyperedge_matrix=hyperedge_matrixt, point_hyperedge_index=point_hyperedge_indext, centers=centerst)
        
        f_gs=self.nets(f_es_embed,hyperedge_matrix=hyperedge_matrixt, point_hyperedge_index=point_hyperedge_indext, centers=centerst)
        
        
        # f_gt_normalized = f_gt / f_gt.norm(p=2)  
        # f_gs_normalized = f_gs / f_gs.norm(p=2)  
        # print(f"f_gt_normalizedmax:{f_gt_normalized.max()}")
        # print(f"f_gs_normalizedmin:{f_gt_normalized.min()}")
        # f_gt=f_gt.reshape(1,f_gt.shape[2],f_gt.shape[1],1)
        # f_gs=f_gs.reshape(1,f_gs.shape[2],f_gs.shape[1],1)
        # ## node-wise contrastive loss
        # f_gt = f_gt.squeeze()
        # f_gs = f_gs.squeeze()
        # print(f"f_gt_shape:{f_gt.shape}")
        # print(f"f_gs_shape:{f_gs.shape}")
        
        f_gt=f_gt.reshape(1,f_gt.shape[2],f_gt.shape[1],1)
        f_gs=f_gs.reshape(1,f_gs.shape[2],f_gs.shape[1],1)
        ## node-wise contrastive loss
        print(f"f_gt_max_beforenorm:{f_gt.max()}")
        print(f"f_gt_mean_beforenorm:{f_gt.mean()}")
        print(f"f_gt_min_beforenorm:{f_gt.min()}")
       
        f_gt = F.normalize(f_gt.squeeze(), p=2, dim=-1)  
        f_gs = F.normalize(f_gs.squeeze(), p=2, dim=-1)  
        print(f"f_gt_max_afternorm:{f_gt.max()}")
        print(f"f_gt_mean_afternorm:{f_gt.mean()}")
        print(f"f_gt_min_afternorm:{f_gt.min()}")
        
        gs_pos = torch.einsum('nc,nc->n', [f_gt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_gs, f_gt]).unsqueeze(-1)
        # print(f"gs_posshape:{gs_pos.shape}")
        # print(f"gt_posshape:{gt_pos.shape}")

        f_gt_reshape = f_gt.view(batch_dim_for_bmm,-1,256).contiguous()
        f_gs_reshape = f_gs.view(batch_dim_for_bmm,-1,256).contiguous()
        # print(f"f_gt_reshapeshape:{f_gt_reshape.shape}")
        # print(f"f_gs_reshapeshape:{f_gs_reshape.shape}")
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))

        import math
        with torch.no_grad():
            mu = 0.7
            sigma = 1
            print("Min value of gs_neg:", gs_neg.min().item())
            print("Mean value of gs_neg:", gs_neg.mean().item())
            print("Max value of gs_neg:",gs_neg.max().item())
            
           
            
            # epsilon = 1e-6 
            # print(f"(l_neg_curbatch) - mu)max:{(- ((l_neg_curbatch) - mu) ) .max().item()}")
            # print(f"(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)))max:{(- ((l_neg_curbatch) - mu) ** 2/ (2 * sigma ** 2)) .max().item()}")
            # print(f"(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)))min:{(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)).min().item()}")
            weight_gs = 1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * \
         torch.exp(- (gs_neg - mu) ** 2 / (2 * sigma ** 2))

            weight_gs = weight_gs / weight_gs.mean(dim=-1, keepdim=True)  
            
            
            weight_gt = 1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * \
         torch.exp(- (gt_neg - mu) ** 2 / (2 * sigma ** 2))

            weight_gt = weight_gt / weight_gt.mean(dim=-1, keepdim=True)  
        #     print(f"1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))):{1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi)))}")
        print(f"weight_gt:{weight_gt.max().item()}")
        print(f"weight_gt:{weight_gt.min().item()}")
       

        weighted_negs = gs_neg * weight_gs.detach()
        weighted_negt = gt_neg * weight_gs.detach()
        diagonal = torch.eye(num_patches, device=device, dtype=torch.bool)[None, :, :]
        weighted_negs.masked_fill_(diagonal, -10.0)
        weighted_negt.masked_fill_(diagonal, -10.0)

        gs_neg = weighted_negs.view(-1, num_patches)
        gt_neg = weighted_negt.view(-1, num_patches)
        


        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()
        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()
        # print(f"out_gt:{out_gt}")
        # print(f"out_gs:{out_gs}")

        loss_gs = self.criterion(out_gs)
        loss_gt = self.criterion(out_gt)
        loss_g = loss_gs + loss_gt

        return loss_g, f_gs, f_gt, adj_s, adj_t   
    
    
    
    def calc_hgnn_pair(self,f_es_embed, f_et_embed, num_patches,  adj_s=None, adj_t=None):
        batch_dim_for_bmm = 1
        T = 0.07
        device = f_es_embed.device
        
          N, C = f_et_embed.shape
        # print(C)
        f_et_embed = f_et_embed.reshape(1, C, -1, 1).contiguous().to(device)
        f_es_embed = f_es_embed.reshape(1, C, -1, 1).contiguous().to(device)
        # print(f"f_et_embed_shape:{f_et_embed.shape}")
        # print(f"f_es_embed_shape:{f_es_embed.shape}")
        
        
        hyperedge_matrixt, point_hyperedge_indext, centerst = construct_hyperedges(f_et_embed, num_clusters=self.k)
        hyperedge_matrixt=hyperedge_matrixt.to(device)
        point_hyperedge_indext=point_hyperedge_indext.to(device)
        centerst=centerst.to(device)
        # hyperedge_matrixs,point_hyperedge_indexs, centerss= construct_hyperedges(f_es_embed.detach(), num_clusters=self.k)
        # hyperedge_matrixs=hyperedge_matrixs.to(device)
        # point_hyperedge_indexs=point_hyperedge_indexs.to(device)
        # centerss=centerss.to(device)
       
        # print(f"centerst:{centerst.max()}")
        f_gt=self.nett(f_et_embed,hyperedge_matrix=hyperedge_matrixt, point_hyperedge_index=point_hyperedge_indext, centers=centerst)
        
        f_gs=self.nets(f_es_embed,hyperedge_matrix=hyperedge_matrixt, point_hyperedge_index=point_hyperedge_indext, centers=centerst)
        
        
        # f_gt_normalized = f_gt / f_gt.norm(p=2)  
        # f_gs_normalized = f_gs / f_gs.norm(p=2)  
        # print(f"f_gt_normalizedmax:{f_gt_normalized.max()}")
        # print(f"f_gs_normalizedmin:{f_gt_normalized.min()}")
        # f_gt=f_gt.reshape(1,f_gt.shape[2],f_gt.shape[1],1)
        # f_gs=f_gs.reshape(1,f_gs.shape[2],f_gs.shape[1],1)
        # ## node-wise contrastive loss
        # f_gt = f_gt.squeeze()
        # f_gs = f_gs.squeeze()
        # print(f"f_gt_shape:{f_gt.shape}")
        # print(f"f_gs_shape:{f_gs.shape}")
        
        f_gt=f_gt.reshape(1,f_gt.shape[2],f_gt.shape[1],1)
        f_gs=f_gs.reshape(1,f_gs.shape[2],f_gs.shape[1],1)
        ## node-wise contrastive loss
        print(f"f_gt_max_beforenorm:{f_gt.max()}")
        print(f"f_gt_mean_beforenorm:{f_gt.mean()}")
        print(f"f_gt_min_beforenorm:{f_gt.min()}")
       
        f_gt = F.normalize(f_gt.squeeze(), p=2, dim=-1) 
        f_gs = F.normalize(f_gs.squeeze(), p=2, dim=-1)  
        print(f"f_gt_max_afternorm:{f_gt.max()}")
        print(f"f_gt_mean_afternorm:{f_gt.mean()}")
        print(f"f_gt_min_afternorm:{f_gt.min()}")
        
        gs_pos = torch.einsum('nc,nc->n', [f_gt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_gs, f_gt]).unsqueeze(-1)
        # print(f"gs_posshape:{gs_pos.shape}")
        # print(f"gt_posshape:{gt_pos.shape}")

        f_gt_reshape = f_gt.view(batch_dim_for_bmm,-1,256).contiguous()
        f_gs_reshape = f_gs.view(batch_dim_for_bmm,-1,256).contiguous()
        # print(f"f_gt_reshapeshape:{f_gt_reshape.shape}")
        # print(f"f_gs_reshapeshape:{f_gs_reshape.shape}")
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))

        import math
        with torch.no_grad():
            mu = 0.3
            sigma = 1
            print("Min value of gs_neg_pair:", gs_neg.min().item())
            print("Mean value of gs_neg_pair:", gs_neg.mean().item())
            print("Max value of gs_neg_pair:",gs_neg.max().item())
            
           
         
            # epsilon = 1e-6 
            # print(f"(l_neg_curbatch) - mu)max:{(- ((l_neg_curbatch) - mu) ) .max().item()}")
            # print(f"(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)))max:{(- ((l_neg_curbatch) - mu) ** 2/ (2 * sigma ** 2)) .max().item()}")
            # print(f"(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)))min:{(- ((l_neg_curbatch) - mu) ** 2 / (2 * sigma ** 2)).min().item()}")
            weight_gs = 1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * \
         torch.exp(- (gs_neg - mu) ** 2 / (2 * sigma ** 2))

            weight_gs = weight_gs / weight_gs.mean(dim=-1, keepdim=True)  
            
            
            weight_gt = 1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * \
         torch.exp(- (gt_neg - mu) ** 2 / (2 * sigma ** 2))

            weight_gt = weight_gt / weight_gt.mean(dim=-1, keepdim=True) 
        #     print(f"1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))):{1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi)))}")
        print(f"weight_gt:{weight_gt.max().item()}")
        print(f"weight_gt:{weight_gt.min().item()}")
        

        weighted_negs = gs_neg * weight_gs.detach()
        weighted_negt = gt_neg * weight_gs.detach()
        diagonal = torch.eye(num_patches, device=device, dtype=torch.bool)[None, :, :]
        weighted_negs.masked_fill_(diagonal, -10.0)
        weighted_negt.masked_fill_(diagonal, -10.0)

        gs_neg = weighted_negs.view(-1, num_patches)
        gt_neg = weighted_negt.view(-1, num_patches)
        


        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()
        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()
        # print(f"out_gt:{out_gt}")
        # print(f"out_gs:{out_gs}")

        loss_gs = self.criterion(out_gs)
        loss_gt = self.criterion(out_gt)
        loss_g = loss_gs + loss_gt

        return loss_g, f_gs, f_gt, adj_s, adj_t 
    
    
    
    
    
    def calc_gnn(self, f_es, f_et, num_patches, gnn=None, adj_s=None, adj_t=None):
        batch_dim_for_bmm=1
        T = 0.07

        ## input features
        G_pos_t,adj_t = nonzero_graph(f_et.detach(), self.nonzero_th,exist_adj=adj_t)
        G_pos_t = G_pos_t.to(f_es.device)
        G_pos_t = dgl.add_self_loop(G_pos_t)
        G_pos_t.ndata['h'] = f_et
        f_gt = gnn(G_pos_t)
        f_gt = f_gt.detach()

        ## output features
        G_pos_s,adj_s = nonzero_graph(f_es.detach(), self.nonzero_th, exist_adj=adj_t)
        G_pos_s = G_pos_s.to(f_es.device)
        G_pos_s = dgl.add_self_loop(G_pos_s)
        G_pos_s.ndata['h'] = f_es
        f_gs = gnn(G_pos_s)         ## shared GNN


        ## node-wise contrastive loss
        f_gt = f_gt.squeeze()
        f_gs = f_gs.squeeze()
        gs_pos = torch.einsum('nc,nc->n', [f_gt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_gs, f_gt]).unsqueeze(-1)
        

        f_gt_reshape = f_gt.view(batch_dim_for_bmm,-1,self.nc)
        f_gs_reshape = f_gs.view(batch_dim_for_bmm,-1,self.nc)
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))#.squeeze()
        
        diagonal = torch.eye(num_patches, device=f_es.device, dtype=torch.bool)[None, :, :]
        gs_neg.masked_fill_(diagonal, -10.0)
        gt_neg.masked_fill_(diagonal, -10.0)

        gs_neg = gs_neg.view(-1, num_patches)
        gt_neg = gt_neg.view(-1, num_patches)
        # print(f"gt_neg.shape{gt_neg.shape}")
        # print(f"gs_neg.shape{gs_neg.shape}")

        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()
        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()

        loss_gs = self.criterion(out_gs)
        loss_gt = self.criterion(out_gt)
        loss_g = loss_gs + loss_gt

        return loss_g, f_gs, f_gt, adj_s, adj_t

    def forward(self, feat_s, feat_t, num_patches=32,patch_ids=None, cams=None):
        return_ids_hard = []
        return_ids_easy = []
        return_feats = []
        loss_g_total = 0
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feat_s)
        for mlp_id, (fs, ft) in enumerate(zip(feat_s, feat_t)):
            # print(f"fs:{fs.max()}")
            # print(f"ft:{ft.min()}")
            epoch_start_time = time.time() 
            print(f"mlp_id:{mlp_id}")
            feat_b_list = []  
            feat_b_list_hard = []
            feat_b_list_easy = []
            id_b_list_hard = []  
            id_b_list_easy = []  
            feat_bt_list_hard = []
            feat_bt_list_easy = []
            loss_g = 0
            gnn = getattr(self, 'gnn_%d' % mlp_id)
            embed = getattr(self, 'embed_%d' % mlp_id)
            pools = getattr(self, 'pools_%d' % mlp_id)
            gnn_pools = getattr(self, 'gnn_pools_%d' % mlp_id)
            B, H, W = fs.shape[0], fs.shape[2], fs.shape[3]
            fs_reshape = fs.permute(0, 2, 3, 1).flatten(1, 2)  # B, L, D
            ft_reshape = ft.permute(0, 2, 3, 1).flatten(1, 2)
            
           
            

            if num_patches > 0:
                
                if patch_ids is not None:
                    for b, patch_id in enumerate(patch_ids[mlp_id]):
                        feat_b_list.append(fs_reshape[b, patch_id, :])
                    fs_reshape_hard = torch.cat(feat_b_list, dim=0).unsqueeze(0)
                    fs_reshape_easy = torch.cat(feat_b_list, dim=0).unsqueeze(0)
                else:
                    num_points = int(min(num_patches, H * W))
                    # TODO: when bs > 1, the speed will be slow.
                    for b in range(cams[mlp_id].shape[0]):
                        cam_b = cams[mlp_id][b]

                        points_sampled = torch.randperm(H * W, device=cam_b.device)[:num_points * self.oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)
                        
                        random_num_points = int(num_points * self.random_ratio)
                        bad_num_points = num_points - random_num_points
                        bad_idx = points_sampled[indices[:bad_num_points]]
                        bad_values = cam_b.flatten()[bad_idx]
                        print(f"cam_b.mean{cam_b.mean()}")
                        
                        mean_valuebad = bad_values.mean()

                        
                        print(f"mean_valuebad：{mean_valuebad}")
                        good_idx = points_sampled[indices[-bad_num_points:]]
                        good_values = cam_b.flatten()[good_idx]

                        
                        mean_valuegood = good_values.mean()

                        
                        print(f"mean_valuegood：{mean_valuegood}")

                        if random_num_points > 0:
                            points_random_hard = torch.randperm(H * W, device=cam_b.device)[-random_num_points:]
                            points_random_easy = torch.randperm(H * W, device=cam_b.device)[-random_num_points:]
                            # points_random = points_sampled[indices[-random_num_points:]]
                            patch_id_hard = torch.cat([good_idx, points_random_hard], dim=0)
                            patch_id_easy = torch.cat([bad_idx, points_random_easy], dim=0)
                            
                        else:
                            patch_id_hard = good_idx
                            patch_id_easy = bad_idx
                           
                        feat_b_list_hard.append(fs_reshape[b, patch_id_hard, :])
                        id_b_list_hard.append(patch_id_hard)
                        feat_b_list_easy.append(fs_reshape[b, patch_id_easy, :])
                        id_b_list_easy.append(patch_id_easy)
                    fs_reshape_hard = torch.cat(feat_b_list_hard, dim=0).unsqueeze(0)
                    fs_reshape_easy = torch.cat(feat_b_list_easy, dim=0).unsqueeze(0)
                xs_sample_hard = fs_reshape_hard.flatten(0, 1)  # reshape(-1, x.shape[1])
                xs_sample_easy = fs_reshape_easy.flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                xs_sample_hard = fs_reshape_hard
                xs_sample_easy = fs_reshape_easy
                id_b_list_hard = []
                id_b_list_easy = []
            
           
            return_ids_hard.append(id_b_list_hard)
            return_ids_easy.append(id_b_list_easy)
          
            
            
            # xs_sample = self.l2norm(xs_sample)
            # print(f"xs_sample:{xs_sample.max()}")
            feat_bt_list_hard = [] 
            for b, patch_id in enumerate(return_ids_hard[mlp_id]):
                feat_bt_list_hard.append(ft_reshape[b, patch_id, :])                 
            ft_reshape_hard = torch.cat(feat_bt_list_hard, dim=0).unsqueeze(0)
            xt_sample_hard = ft_reshape_hard.flatten(0, 1)  # reshape(-1, x.shape[1])
            
            feat_bt_list_easy = []  
            for b, patch_id in enumerate(return_ids_easy[mlp_id]):
                feat_bt_list_easy.append(ft_reshape[b, patch_id, :])                 
            ft_reshape_easy = torch.cat(feat_bt_list_easy, dim=0).unsqueeze(0)
            xt_sample_easy = ft_reshape_easy.flatten(0, 1)  # reshape(-1, x.shape[1])
            
            f_et_embed_easy = xt_sample_easy
            f_es_embed_easy = xs_sample_easy
            f_et_embed_hard = xt_sample_hard
            f_es_embed_hard = xs_sample_hard
            
            ### graph loss before pooling
            loss_gnn, gnn_s, gnn_t, adj_s, adj_t = self.calc_hgnn(f_es_embed_hard, f_et_embed_hard,num_patches)
            
            loss_gnn2, _,_,_,_, = self.calc_hgnn_pair(f_es_embed_easy, f_et_embed_easy,num_patches)
            
            loss_gnn=loss_gnn+loss_gnn2
            
            ### pooling
            f_es = gnn_s
            f_et = gnn_t
            loss_g += loss_gnn
            if self.pooling_num > 0:
                pool_f_es = f_es
                pool_f_et = f_et
                for pooling in range(self.pooling_num):
                    downscale = self.down_scale * (2 ** pooling)

                    pool_f_et, pool_f_es, pool_adj_t = pools[pooling] \
                        (pool_f_et, pool_f_es, adj_t, num_patches // downscale)
                    loss_gnn, pool_f_es, pool_f_et, adj_s, adj_t = self.calc_gnn(pool_f_es, pool_f_et,
                                                                                 num_patches // downscale,
                                                                                 gnn=gnn_pools[pooling], adj_s=None,
                                                                                 adj_t=pool_adj_t)
                    adj_t = pool_adj_t
                   
                    level = int(pooling + 1)
                    loss_g += loss_gnn * self.pooling_ratio[level]

                    

            loss_g /= (self.pooling_num + 1)
            loss_g_total += loss_g
        loss_g_total /= len(feat_s)

        return loss_g_total



class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hop):
        super(Encoder, self).__init__()
        self.conv1 = TAGConv(in_dim, hidden_dim, k=num_hop)
        self.l2norm = Normalize(2)

    def forward(self, g, edge_weight=None):
        h = g.ndata['h']
        h = self.l2norm(self.conv1(g, h, edge_weight))

        return h


class Pool(nn.Module):

    def __init__(self,  in_dim):
        super(Pool, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        
    def forward(self, ht, hs, g, k,):
        '''

        :param ht: teacher feature
        :param hs: student feature
        :param g: teacher adjacent matrix
        :param k: Number of pooled node
        :return: pooled teacher feat, pooled student feat, pooled adjacency matrix
        '''

        g_coo = g.tocoo()
        g_data = torch.sparse.LongTensor(torch.LongTensor([g_coo.row.tolist(), g_coo.col.tolist()]),
                              torch.LongTensor(g_coo.data.astype(np.int32))).to_dense()

        ## scores for each node
        weights = self.proj(ht).squeeze()
        scores = self.sigmoid(weights)

        ## pool graphs by scores
        g, new_ht, idx = top_k_graph(scores, g_data, ht, k)
        _, new_hs, _ = top_k_graph(scores, g_data, hs, k)
        return new_ht, new_hs, g


