import os
import torch.nn as nn
import unet_model
import rednet

F_DIR = os.path.split(os.path.abspath(__file__))[0]
PARAM_DIR = os.path.abspath(os.path.join(F_DIR,'..','default_params'))
# Glossary for pre-defined model architecture
def model_glossary():
    gl_d = {
        'unet':unet_model.UNet,
        'rednet10':rednet.REDNet10,
        'rednet20':rednet.REDNet20,
        'rednet30':rednet.REDNet30,
        'dncnn':'',
        'srcnn':'',
    }
    
    return gl_d


def default_param_file_glossary(param_dir=PARAM_DIR):
    return {_k:os.path.join(param_dir,_k+'.json') for _k in model_glossary()}
    
