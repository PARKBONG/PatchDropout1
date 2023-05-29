import torch
import matplotlib.pyplot as plt
import numpy as np

class PatchDropout(torch.nn.Module):
    """ 
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    """
    def __init__(self, keep_rate=0.5, sampling="SVD", token_shuffling=False):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        
        # self.sampling = sampling
        self.sampling = "crop_KR25"
        self.token_shuffling = token_shuffling

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        # print("before")
        # print(x.shape)
        

        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        # print("after")
        # print(x.shape)
        # exit()
        return x
    
    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        
        # elif self.sampling == "random":
        #     return self.random_mask(x)
        
        elif self.sampling == "crop_KR25":
            return self.crop_mask_KR25(x)
        
        # elif self.sampling == "structure":
        #     return self.structure_mask(x)
        
        # elif self.sampling == "SVD":
        #     return self.svd_mask(x)
            
        else:
            return NotImplementedError(f"PatchDropout does ot support {self.sampling} sampling")
    
    def uniform_mask(self, x):
        """
        Uniform: This strategy also samples patches randomly, but it keeps the probability of selection uniform. In other words, each element independently has a chance of keep_rate to be kept.
        """
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        keep = int(_L * self.keep_rate)
        # create uniform mask
        patch_mask = torch.rand(N, _L, device=x.device) < self.keep_rate
    
    # use the mask to select tokens to keep
        patch_mask = torch.nonzero(patch_mask, as_tuple=False)[:, 1]
        patch_mask = patch_mask + 1  # compensate for the CLS token
    
    # if we have more tokens than we should keep, randomly drop some
        if patch_mask.shape[0] > keep:
            drop = patch_mask.shape[0] - keep
            drop_indices = torch.randperm(patch_mask.shape[0])[:drop]
            patch_mask = patch_mask[drop_indices]
    
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
    
        return patch_mask

    # def random_mask(self, x):
    #     pass
    
    def crop_mask_KR25(self, x):
        
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        size = 8
        true_len = 4 # 75%
        rand_index_row = np.random.randint(0, size - true_len + 1) # 0, 1, 2
        rand_index_col = np.random.randint(0, size - true_len + 1) # 0, 1, 2

        false_array = np.full((size, size), False)

        # Create an array filled with True values
        true_array = np.full((true_len, true_len), True)

        false_array[rand_index_row:rand_index_row + true_len, rand_index_col: rand_index_col + true_len] = true_array
        
        patch_mask = false_array.flatten()
        return patch_mask
            
    
    def crop_mask_KR50(self, x):
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        x[:,] 
        pass 
    
    def crop_mask_75(self, x):
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        x[:,] 
        pass 

    def crop_mask_90(self, x):
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        x[:,] 
        pass 
    
    # def structure_mask(self, x):
    #     pass
    
    def svd_mask(self, x):
        """
        Apply SVD and return a patch mask keeping patches with top singular values.
        """
        N, L, D = x.shape
        x_2d = x.view(N*L, D)

        # Compute the SVD
        U, S, V = torch.svd(x_2d)
        
        # print("U shape: ", U.shape)
        # print("S shape: ", S.shape)
        # print("V shape: ", V.shape)

        # Determine the number of patches to keep
        keep = int(L * self.keep_rate)

        # Sort the singular values and keep indices of the top 'keep' values
        _, indices = torch.sort(S, descending=True)
        keep_indices = indices[:keep]

        # Generate the patch mask for each instance in the batch
        patch_mask = keep_indices.view(N, -1)

        return patch_mask