import torch
import torch.nn as nn
import ssim_loss
import vgg_loss


def loss_glossary():
    gl_d = {
        'mse':nn.MSELoss,
        'mae':nn.L1Loss,
        'vgg':vgg_loss.VGGPerceptualLoss,
        'ssim':ssim_loss.SSIM,
    }
    return gl_d
