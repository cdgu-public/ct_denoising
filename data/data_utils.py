import os
import numpy as np
import matplotlib.pyplot as plt


############################################
################# Data IO ##################
############################################


def save_img(arr,file):
    np.save(file,arr)
    
    
def load_img(file):
    return np.laod(file)


def show_image(arr):
    _data = arr.squeeze()
    plt.imshow(_data,cmap='gray')
    plt.show()

def save_plt(arr,file):
    show_image(arr)
    plt.savefig(file)
    