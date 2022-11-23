#!/usr/bin/env python3
import os
import sys
import argparse
import time
import datetime
import utils
import pipeline

# import loss.losses as losses
# import model.models as models

def main():
    print('Start')
    start = time.time()
    s_dt = datetime.datetime.utcfromtimestamp(start)
    print(f"{s_dt.year}.{s_dt.month}.{s_dt.day} - {s_dt.hour}:{s_dt.minute}:{s_dt.second}")
    
    parser = argparse.ArgumentParser(description='CT denoising')
    subparsers = parser.add_subparsers(title='Job',
                                       description="train or run_model",
                                       dest='job',
                                       help='Job to do: train / denoise'
                                      )
    parser_tr = subparsers.add_parser('train',help='Training')
    
    parser_tr.add_argument('-i','--input',dest='data_dir',help='Input')
    parser_tr.add_argument('-o','--output',dest='output_dir',help='Output')
    parser_tr.add_argument('-m','--model_type',choices=['unet','rednet'],help='Type of model')
    parser_tr.add_argument('-l','--loss',dest='loss_type',
                           choices=['mse','mae','vgg','ssim'],
                           help='Loss type')
    
    
    parser_tr.add_argument('--learning_rate',type=float,dest='lr',default=1e-4,help='Learning rate')
    parser_tr.add_argument('--batch_size',type=int,default=16,help='Batch size')
    parser_tr.add_argument('--workers',type=int,default=4,help='No. of workers for dataloader')
    parser_tr.add_argument('--iterations',type=int,default=1e6,help='Iteration')
    parser_tr.add_argument('--checkpoint',type=str,default=None,help='Checkpoint file if inferrable')
    parser_tr.add_argument('--start_epoch',type=int,default=0,help='Checkpoint file if inferrable')
    parser_tr.add_argument('--early_stop_patience',type=int,default=10,help='early_stop_patience')
    
    
    parser_rn = subparsers.add_parser('denoise',help='Denoise data')
    
    parser_rn.add_argument('-m','--model',dest='checkpoint',help='Model checkpoint file')
    parser_rn.add_argument('-i','--input',dest='data_dir',help='Input')
    parser_rn.add_argument('-l','--loss',dest='loss_type',
                           choices=['mse','mae','vgg','ssim'],
                           help='Loss type')
    parser_rn.add_argument('-o','--output',dest='output_dir',help='Output')
    
    
    args = parser.parse_args()
    
    if args.job == 'train':
        pipeline.run_train(**args.__dict__)
        
    elif args.job == 'denoise':
        pipeline.run_denoise(**args.__dict__)
    
    print('Finished')
    end = time.time()
    e_dt = datetime.datetime.utcfromtimestamp(end)
    print(f"{e_dt.year}.{e_dt.month}.{e_dt.day} - {e_dt.hour}:{e_dt.minute}:{e_dt.second}")
    time_cost = datetime.datetime.utcfromtimestamp(end-start)
    print(f"Time cost : {time_cost.day-1} - {time_cost.hour}:{time_cost.minute}:{time_cost.second}")
    
    
if __name__ == '__main__':
    main()

    