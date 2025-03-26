import torch
from torch import nn
from packaging import version

class PatchNCELoss(nn.Module):
    def __init__(self, nce_includes_all_negatives_from_minibatch, batch_size, nce_T, tau_plus=0.7, beta=0.5):
        super().__init__()
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.tau_plus = tau_plus  
        self.beta = beta  
        self.total_time = 0
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.relu=torch.nn.ReLU()
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         l_neg = l_neg_curbatch.view(-1, npatches)
        # print(f"l_neg_curbatch:{l_neg_curbatch}")
        
        import math
        with torch.no_grad():
            mu = 0.7
            sigma = self.beta

            
           
           
            # epsilon = 1e-6 
                       weight = 1. / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * \
         torch.exp(- (l_neg_curbatch - mu) ** 2 / (2 * sigma ** 2))

            weight = weight / weight.mean(dim=-1, keepdim=True)  
        weighted_neg = l_neg_curbatch * weight.detach()
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        weighted_neg.masked_fill_(diagonal, -10.0)
        l_neg = weighted_neg.view(-1, npatches)
       
                out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        
    
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
                return loss

# class PatchNCELoss(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

#     def forward(self, feat_q, feat_k):
#         num_patches = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(
#             feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
#         l_pos = l_pos.view(num_patches, 1)

#         # neg logit

#         # Should the negatives from the other samples of a minibatch be utilized?
#         # In CUT and FastCUT, we found that it's best to only include negatives
#         # from the same image. Therefore, we set
#         # --nce_includes_all_negatives_from_minibatch as False
#         # However, for single-image translation, the minibatch consists of
#         # crops from the "same" high-resolution image.
#         # Therefore, we will include the negatives from the entire minibatch.
#         if self.opt.nce_includes_all_negatives_from_minibatch:
#             # reshape features as if they are all negatives of minibatch of size 1.
#             batch_dim_for_bmm = 1
#         else:
#             batch_dim_for_bmm = self.opt.batch_size

#         # reshape features to batch size
#         feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
#         feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
#         npatches = feat_q.size(1)
#         l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

#         # diagonal entries are similarity between same features, and hence meaningless.
#         # just fill the diagonal with very small number, which is exp(-10) and almost zero
#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         l_neg = l_neg_curbatch.view(-1, npatches)

#         out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))

#         return loss

    
    
# from packaging import version
# import torch
# from torch import nn

# class PatchNCELoss(nn.Module):
#     def __init__(self, nce_includes_all_negatives_from_minibatch, batch_size, nce_T):
#         super().__init__()
#         self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
#         self.batch_size = batch_size
#         self.nce_T = nce_T
        
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

#     def forward(self, feat_q, feat_k):
        
#         num_patches = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(
#             feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
#         l_pos = l_pos.view(num_patches, 1)

#         # neg logit
#         if self.nce_includes_all_negatives_from_minibatch:
#             batch_dim_for_bmm = 1
#         else:
#             batch_dim_for_bmm = self.batch_size

#         feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
#         feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
#         npatches = feat_q.size(1)
#         l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         l_neg = l_neg_curbatch.view(-1, npatches)

#         out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
#         # epoch_end_time = time.time()  
#         # epoch_time = epoch_end_time - epoch_start_time  
#         # self.total_time += epoch_time  
#     
#         return loss












































# # from packaging import version
# # import torch
# # from torch import nn

# import time
# # class PatchNCELoss(nn.Module):
# #     def __init__(self, opt):
# #         super().__init__()
# #         self.opt = opt
# #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
# #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

# #     def forward(self, feat_q, feat_k):
# #         num_patches = feat_q.shape[0]
# #         dim = feat_q.shape[1]
# #         feat_k = feat_k.detach()

# #         # pos logit
# #         l_pos = torch.bmm(
# #             feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
# #         l_pos = l_pos.view(num_patches, 1)

# #         # neg logit

# #         # Should the negatives from the other samples of a minibatch be utilized?
# #         # In CUT and FastCUT, we found that it's best to only include negatives
# #         # from the same image. Therefore, we set
# #         # --nce_includes_all_negatives_from_minibatch as False
# #         # However, for single-image translation, the minibatch consists of
# #         # crops from the "same" high-resolution image.
# #         # Therefore, we will include the negatives from the entire minibatch.
# #         if self.opt.nce_includes_all_negatives_from_minibatch:
# #             # reshape features as if they are all negatives of minibatch of size 1.
# #             batch_dim_for_bmm = 1
# #         else:
# #             batch_dim_for_bmm = self.opt.batch_size

# #         # reshape features to batch size
# #         feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
# #         feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
# #         npatches = feat_q.size(1)
# #         l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

# #         # diagonal entries are similarity between same features, and hence meaningless.
# #         # just fill the diagonal with very small number, which is exp(-10) and almost zero
# #         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
# #         l_neg_curbatch.masked_fill_(diagonal, -10.0)
# #         l_neg = l_neg_curbatch.view(-1, npatches)

# #         out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

# #         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
# #                                                         device=feat_q.device))

# #         return loss

    
    
# from packaging import version
# import torch
# from torch import nn

# class PatchNCELoss(nn.Module):
#     def __init__(self, nce_includes_all_negatives_from_minibatch, batch_size, nce_T):
#         super().__init__()
#         self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
#         self.batch_size = batch_size
#         self.nce_T = nce_T
#         self.total_time=0
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

#     def forward(self, feat_q, feat_k):
#         epoch_start_time = time.time()
#         num_patches = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(
#             feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
#         l_pos = l_pos.view(num_patches, 1)

#         # neg logit
#         if self.nce_includes_all_negatives_from_minibatch:
#             batch_dim_for_bmm = 1
#         else:
#             batch_dim_for_bmm = self.batch_size

#         feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
#         feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
#         npatches = feat_q.size(1)
#         l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         l_neg = l_neg_curbatch.view(-1, npatches)

#         out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
#         # epoch_end_time = time.time()  # 记录每个 epoch 的结束时间
#         # epoch_time = epoch_end_time - epoch_start_time  # 计算当前 epoch 的时间
#         # self.total_time += epoch_time  # 累加到总时间
#         # print(f"nce_time:{self.total_time}")
#         return loss
