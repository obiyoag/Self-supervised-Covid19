import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.autograd import Variable
from torch.nn import MSELoss
from torchvision.models import vgg19



class DiceLoss(nn.Module):
    def __init__(self, smooth=1, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, labels):
        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / (torch.sum(preds) + torch.sum(labels) + self.smooth)

class log_loss(nn.Module):
    def __init__(self):
        super(log_loss, self).__init__()        
    def forward(self, logits, label, smooth = 1.):
        area_union = torch.sum(logits * label, dim = (0,2,3), keepdim = True)
        area_logits = torch.sum(logits, dim = (0,2,3), keepdim = True)
        area_label = torch.sum(label, dim = (0,2,3), keepdim = True)
        in_dice = torch.mean(torch.pow((-1) * torch.log((2 * area_union + 1e-7)/(area_logits + area_label + smooth)), 0.3))
        return in_dice


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGG_FeatureExtractor().to("cuda")
        self.p_criterion = nn.L1Loss()

    def forward(self, x, y):
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

class DenoiseLoss(nn.Module):
    def __init__(self, tv_weight, p_weight):
        super(DenoiseLoss,self).__init__()
        self.tv_weight = tv_weight
        self.p_weight = p_weight
        self.tvloss = TVLoss()
        self.ploss = PerceptualLoss()
    def forward(self,output, target):
        return self.tv_weight*self.tvloss(output) + self.p_weight*self.ploss(output, target)