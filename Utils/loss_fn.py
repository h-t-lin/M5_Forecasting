import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def loss_settings(loss_type='MSE', reduction='mean', *args, **kwargs):
    '''
    ### Return loss functions defined in...  
    * torch.nn: "MSE", "L2", "MAE", "L1", "BCE", "CE", "CrossEntropy"  
    * Customized_Loss: "L1+L2", "MW-SSIM", "DSSIM"  
    ### 
    reduction: str = ( "none" | "mean" | "sum" )
    '''
    torch_loss_options = ['MSE', 'L2', 'MAE', 'L1', 'BCE', 'CE', 'CrossEntropy']
    customized_loss_options = ['L1+L2', 'MW-SSIM', 'DSSIM', 'DSSIM_L1_L2_Mix']

    # from torch.nn
    if loss_type in torch_loss_options[0:2]:
        return nn.MSELoss(reduction=reduction)
    elif loss_type in torch_loss_options[2:4]:
        return nn.L1Loss(reduction=reduction)
    elif loss_type==torch_loss_options[4]:
        return nn.BCELoss(reduction=reduction, **kwargs)
    elif loss_type in torch_loss_options[5:7]:
        return nn.CrossEntropyLoss(reduction=reduction, **kwargs)
    # from Customized_Loss
    elif loss_type==customized_loss_options[0]:
        return L1_L2_Addition(reduction=reduction, **kwargs)
    elif loss_type==customized_loss_options[1]:
        return Mask_Weighted_SSIM(reduction=reduction, **kwargs)
    elif loss_type==customized_loss_options[2]:
        return DSSIM(reduction=reduction, **kwargs)
    elif loss_type==customized_loss_options[3]:
        return DSSIM_L1_L2_Mix(reduction=reduction, **kwargs)
    else:
        raise TypeError(f'There is no loss function named {loss_type}.\nPlease choose a function from {torch_loss_options} or {customized_loss_options}.')


'''----Customized Loss Functions----'''
class Customized_Loss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        '''**by NTU MMIO LDTAU Proj.**'''
        super().__init__(None, None, reduction)

    def _setConfig_(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
    
    def _setFloatAttr_(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, str):
                try: setattr(self, key, float(val))
                except: raise(ValueError(f"Value of <{key}>:{val} cannot be converted to type float."))
            elif isinstance(val, (float, int)): setattr(self, key, val)
            else: raise(TypeError(f"Type of <{key}> has to be (str, int, float), but <{type(val)}>"))

    def Loss_Reduction(self, loss: Tensor):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        
    def image_size_equality(self, input, target) -> bool:
        if not type(input)==type(target): raise TypeError(f"Types of input and target must be the same. But got {(type(input), type(target))}")
        if isinstance(input, Tensor):
            return input.size() == target.size()
        elif isinstance(input, np.ndarray):
            return input.shape == target.shape
        else: raise TypeError(f"Inputs type must be torch.Tensor OR numpy.ndarray, but got {type(input)}")

    def Tensor2Numpy(self, tensor: Tensor):
        if not isinstance(tensor, Tensor):
            return tensor
        return tensor.detach().cpu().numpy()


class L1_L2_Addition(Customized_Loss):
    def __init__(self, alpha=0.5, reduction: str = 'mean', **kwargs) -> None:
        '''
        ## L1+L2 Loss
        alpha determines the portion of L1(MAE)  
        ```
        alpha*L1 + (1-alpha)*L2
        ```
        '''
        super().__init__(reduction)
        self._setFloatAttr_(alpha=alpha)
        if not 0<=self.alpha<=1: raise ValueError("alpha must between [0,1]")

    def forward(self, input: Tensor, target: Tensor):

        MAE = loss_settings('MAE', reduction='none')
        MSE = loss_settings('MSE', reduction='none')

        loss = self.alpha*MAE(input, target) + (1-self.alpha)*MSE(input, target)

        return self.Loss_Reduction(loss)

class Mask_Weighted_SSIM(Customized_Loss):
    def __init__(self, win_size=5, data_range=1, reduction: str = 'mean', **kwargs) -> None:
        '''
        SSIM weighted by mask.  
        Value:  
        > 0 if mask is zero matrix  
        > float in (...,1], otherwise  
        '''
        super().__init__(reduction)
        self.win_size = win_size
        self.data_range = data_range

    def forward(self, input: Tensor, target: Tensor, mask=None):
        
        # Calculate loss by metric.
        metric = SSIM(kernel_size=self.win_size, reduction='none', data_range=self.data_range, return_full_image=True)
        loss, loss_map = metric(input, target)

        # If no mask, return loss without being weighted.
        if mask==None:
            return self.Loss_Reduction(loss)
        
        # Check mask size.
        assert self.image_size_equality(mask, input)
        
        # Calculate weighted loss by applying mask to the local SSIM map.
        total_weights = mask.sum(dim=(1,2,3))
        weighted_loss = (loss_map * mask).sum(dim=(1,2,3)) / (torch.max(total_weights, torch.ones_like(total_weights)))

        return self.Loss_Reduction(weighted_loss)

class DSSIM(Customized_Loss):
    def __init__(self, win_size=5, data_range=1, C0=1, reduction: str = 'mean', **kwargs) -> None:
        '''
        ## Structural Dissimilarity Index Measure  
        Suggestion: set C in [0.5, 5]  
        [Loss Curve Visualized by Desmos](https://www.desmos.com/calculator/szw0gsoq0r)
        '''
        super().__init__(reduction)
        self.win_size = win_size
        self.data_range = data_range
        if not C0>0: raise ValueError("C must >0.")
        self.C = C0*.1

    def forward(self, input: Tensor, target: Tensor):
        
        # Calculate loss by metric.
        metric = SSIM(kernel_size=self.win_size, reduction='none', data_range=self.data_range)
        ssim = metric(input, target)

        loss = self.C / (ssim-1-self.C) + 1
        return self.Loss_Reduction(loss)

class DSSIM_L1_L2_Mix(Customized_Loss):
    def __init__(self, alpha=0.5, beta=10.0, reduction: str = 'mean', **kwargs) -> None:
        '''
        ## DSSIM * (L1 + L2) Loss
        alpha determines the portion of L1(MAE)  
        ```
        l1l2 := alpha*L1 + (1-alpha)*L2
        loss := beta*DSSIM*l1l2
        ```
        '''
        super().__init__(reduction)
        self._setFloatAttr_(alpha=alpha, beta=beta)
        if not 0<=self.alpha<=1: raise ValueError("alpha must between [0,1]")

    def forward(self, input: Tensor, target: Tensor):

        MAE = loss_settings('MAE', reduction='none')
        MSE = loss_settings('MSE', reduction='none')
        DS = loss_settings('DSSIM', reduction='none')

        loss = (self.alpha*MAE(input, target) + (1-self.alpha)*MSE(input, target)).mean(dim=tuple(range(1, input.ndim)))
        loss = self.beta*DS(input, target)*loss

        return self.Loss_Reduction(loss)

