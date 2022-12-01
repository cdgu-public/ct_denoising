#!usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


center_coord = (192,192) # (0~511,0~511)
crop_size = 64


def crop_npy(npy_data,center_coord=center_coord,crop_size=crop_size):
    xmin = int(center_coord[0]-(crop_size/2))
    xmax = xmin + crop_size
    ymin = int(center_coord[1]-(crop_size/2))
    ymax = ymin + crop_size
    plt_coord = npy_data[xmin:xmax,ymin:ymax]
    return plt_coord


def save_img(npy,out_f='./output.png'):
    _curr_img = npy.squeeze()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(_curr_img, cmap='gray')
    plt.savefig(out_f)

    
def main():
    parser = argparse.ArgumentParser(description='Crop image')
    parser.add_argument('-i','--input',type=str,help='Input dir')
    parser.add_argument('-o','--output',type=str,help='Output dir')
    parser.add_argument('-x','--x_center',type=float,default=192,help='x_center')
    parser.add_argument('-y','--y_center',type=float,default=192,help='y_center')
    parser.add_argument('-s','--input_size',type=int,default=512,help='input data size')
    parser.add_argument('-c','--crop_size',type=int,default=128,help='crop_size')
    parser.add_argument('-m','--minibatch_no',type=int,default=0,help='No. of data for npy chunk')
    args = parser.parse_args()
    
    center_coord = [0,0]
    
    if args.x_center < 1:
        center_coord[0]=int(args.x_center*args.input_size)
    else:
        center_coord[0]=int(args.x_center)
    if args.y_center < 1:
        center_coord[1]=int(args.y_center*args.input_size)
    else:
        center_coord[1]=int(args.y_center)
    
    os.makedirs(args.output,exist_ok=True)
    
    for f_n in os.listdir(args.input):
        input_f = os.path.join(args.input,f_n)
        output_f = os.path.join(args.output,f_n+'.png')
        try:
            npy_data = np.load(input_f)[args.minibatch_no,0,:,:]
            cut_img = crop_npy(
                npy_data=npy_data,
                center_coord=center_coord,
                crop_size=args.crop_size
            )
            save_img(npy=cut_img,out_f=output_f)
        except:
            print(f'Errored:{input_f}')
        
if __name__=='__main__':
    main()
    