import torch.nn as nn
import unet_model
import rednet

def model_glossary():
    gl_d = {
        'unet':unet_model.UNet,
        'rednet':rednet.REDNet10,
    }
    
    return gl_d

