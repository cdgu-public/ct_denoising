import torch
from torch.utils.data import Dataset
import json
import os
import h5py
import numpy as np


class HUTransformer:
    def __init__(self,u_water:float=0.0192):
        self.u_water = u_water
        
    def _transform_(self,data):
        return (data-self.u_water)*1000/self.u_water
    
    def _inv_transform_(self,data):
        return (data/1000*self.u_water)+self.u_water
    
    def __call__(self,data):
        return self._transform_(data)
    


class CTDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_dir:'list or str', hu_transformer=HUTransformer(u_water=0.0192), transformer=None):
        """
        :param data_dir: 
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        """
        if type(data_dir) == str:
            # if data_dir is a specific file or directory
            if os.path.isdir(data_dir):
                self.data_list = [os.path.abspath(os.path.join(data_dir,_f)) for _f in os.listdir(data_dir)]
            elif os.path.isfile(data_dir):
                self.data_list = [os.path.abspath(_f)]
            else: raise(TypeError)
                
        elif type(data_dir) in [list,set,tuple]:
            self.data_list = []
            for _d_ in data_dir:
                if os.path.isfile(_d_):
                    self.data_list.append(os.path.abspath(_d_))
                elif os.path.isdir(_d_):
                    self.data_list.extend([os.path.abspath(os.path.join(_d_,_f)) for _f in os.listdir(_d_)])
        else: raise(TypeError)
        self.hu_transformer = hu_transformer
        self.transformer = transformer
        
        i = 0
        for file_path in self.data_list:
            if i == 0:
                i+=1
                ndct = np.array(h5py.File(file_path)['f_nd'])
                qdct = np.array(h5py.File(file_path)['f_qd'])
            else:
                new_ndct = np.array(h5py.File(file_path)['f_nd'])
                new_qdct = np.array(h5py.File(file_path)['f_qd'])
                ndct = np.concatenate((ndct, new_ndct), axis=0)
                qdct = np.concatenate((qdct, new_qdct), axis=0)
        if self.hu_transformer:
            self.ndct = self.hu_transformer(ndct)
            self.qdct = self.hu_transformer(qdct)
        else:
            self.ndct = ndct
            self.qdct = qdct
        assert self.ndct.shape[0] == self.qdct.shape[0]
        
    def __getitem__(self, i):
        _ndct = self.ndct[i]
        _qdct = self.qdct[i]
        if self.transformer:
            _ndct = self.transformer(_ndct)
            _qdct = self.transformer(_qdct)
        return _ndct, _qdct

    def __len__(self):
        return len(self.ndct)

    

    

    