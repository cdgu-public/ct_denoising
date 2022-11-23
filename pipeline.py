#!/usr/bin/env python3
"""pipeline for training/evaluation"""
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch import nn
import model.models as model_gls #
import loss.losses as loss_gls
import data.dataset as dataset #
from utils import *
import matplotlib.pyplot as plt

cudnn.benchmark = True

def run_train(data_dir,
              model_type='unet',
              loss_type='mse',
              batch_size:int=16,
              checkpoint=None,
              n_channels:int=1,
              iterations=1e6,
              start_epoch:int=0,
              lr:float=1e-4,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              early_stop_patience:int=20,
              checkpoint_step:int=10,
              output_dir:str='./',
              print_freq:int=500,
              workers:int=4,
              grad_clip=None,
#               crop_size=96,
              u_water=0.0192,
              **kwargs,
             ):
    """
    Training.
    """
    
#     global start_epoch, epoch, checkpoint
    print("Training started.")
    # Initialize model or load checkpoint
    ##################### DATA #####################
    # data location
    train_data_loc = os.path.join(data_dir,'train')
    val_data_loc = os.path.join(data_dir,'validation')
    test_data_loc = os.path.join(data_dir,'test')
    # transformers
    hu_transformer = dataset.HUTransformer(u_water=u_water)
    transformer = None
    
    # Custom dataloaders
    train_dataset = dataset.CTDataset(
        data_dir=train_data_loc,hu_transformer=hu_transformer,transformer=transformer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)  

    val_dataset = dataset.CTDataset(
        data_dir=val_data_loc,hu_transformer=hu_transformer,transformer=transformer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)  
    
    test_dataset = dataset.CTDataset(
        data_dir=test_data_loc,hu_transformer=hu_transformer,transformer=transformer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)  
    
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_type=model_type,
        loss_type=loss_type,
        batch_size=batch_size,
        checkpoint=checkpoint,
        n_channels=n_channels,
        iterations=iterations,
        start_epoch=start_epoch,
        lr=lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        early_stop_patience=early_stop_patience,
        output_dir=output_dir,
        grad_clip=grad_clip,
    )

    
def run_denoise(checkpoint,
                data_dir,
                loss_type,
                output_dir,
                workers=4,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                u_water=0.0192,
                **kwargs):
    if not device:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    hu_transformer = dataset.HUTransformer(u_water=u_water)
    transformer=None
    test_dataset = dataset.CTDataset(
        data_dir=data_dir,hu_transformer=hu_transformer,transformer=transformer)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=workers, pin_memory=True)
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    
    run_model(
        checkpoint=checkpoint, loss_type=loss_type,
        output_dir=output_dir, dataloader=dataloader,
        device=device)
    

def train_model(train_loader,
                val_loader,
                test_loader,
                model_type='unet',
                loss_type='mse',
                batch_size:int=16,
                checkpoint=None,
                n_channels:int=1,
                iterations=1e6,
                start_epoch:int=0,
                lr:float=1e-4,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                early_stop_patience:int=None,
                checkpoint_step:int=10,
                output_dir:str='./',
                print_freq:int=500,
                workers:int=4,
                grad_clip=None,
#                 crop_size=96,
               ):
#     global start_epoch, epoch, checkpoint
    print("Training started.")
    start = time.time()
    # Initialize model or load checkpoint
    if checkpoint is None:
        if type(model_type) == str:
            model_cls = model_gls.model_glossary()[model_type]
            if model_type == 'unet':
                model = model_cls(n_channels=n_channels, n_classes=n_channels, bilinear=False)
            elif model_type == 'rednet':
                model = model_cls()
        else:
            model = model_type
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # output_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    ckpt_dir = os.path.join(output_dir,'checkpoint')
    os.makedirs(ckpt_dir,exist_ok=True)
    _val_loss_f = os.path.join(output_dir,'val_losses.txt')
    _val_loss_json = os.path.join(output_dir,'val_losses.json')
    test_result_dir = os.path.join(output_dir,'test_set_results')
    os.makedirs(test_result_dir,exist_ok=True)
    
    # Move to default device
    model = model.to(device)
    loss_fn = loss_gls.loss_glossary()[loss_type]
    criterion = loss_fn().to(device)

    # Total number of epochs to train for
    loss_by_e = {}
    epochs = int(iterations // len(train_loader) + 1)
    curr_loss = float('inf')
    curr_loss_pat = 0
    with open(_val_loss_f,'w') as f:
        f.write('epoch\tvalidation_loss\tepoch_time\n')
    # Epochs
    for epoch in range(start_epoch, epochs):
        e_start = time.time()
        # One epoch's training
        training_step(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   f'{ckpt_dir}/checkpoint.{epoch:04d}.pth.tar')
        # validation
        with torch.no_grad():
            val_loss = evaluation_step(dataloader=val_loader,model=model,criterion=criterion,device=device)
            print(f"{epoch}:\t{val_loss.avg}")
            
            loss_by_e[epoch] = val_loss.avg
            if val_loss.avg > curr_loss:
                curr_loss_pat += 1
            else:
                curr_loss = val_loss.avg
                curr_loss_pat = 0
                
            with open(_val_loss_f,'a') as f:
                f.write(f"{epoch}\t{val_loss.avg}\t{time.time()-e_start:.3f}\n")
            if type(early_stop_patience) == int:
                if not epoch%checkpoint_step and epoch > early_stop_patience and epoch > checkpoint_step:
                    os.system(f'rm {ckpt_dir}/checkpoint.{epoch-early_stop_patience:04d}.pth.tar')
                if curr_loss_pat > early_stop_patience:
                    print('Early stopped at epoch %s'%epoch)
                    break
    # Pickup the best model
    epoch_ser = pd.Series(loss_by_e)
    min_loss_epoch = epoch_ser.index[np.argmin(epoch_ser)]
    os.system(f'cp {ckpt_dir}/checkpoint.{min_loss_epoch:04d}.pth.tar {output_dir}/checkpoint.best_model.{min_loss_epoch:04d}.pth.tar')
    
    try:
        with open(_val_loss_json,'wb') as f:
            f.write(json.dumps(loss_by_e).encode())
    except:
        print(loss_by_e)
        
    # result for test set
    
    test_loss = run_model(
        checkpoint=f'{output_dir}/checkpoint.best_model.{min_loss_epoch:04d}.pth.tar',
        loss_type=loss_type,
        output_dir=test_result_dir,
        dataloader=test_loader,
        device=device,
    )
    print(test_loss)
    
    
def run_model(checkpoint,
              output_dir,
              dataloader,
              loss_type='mse',
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             ):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    
    model = model.to(device)
    
    loss_fn = loss_gls.loss_glossary()[loss_type]
    criterion = loss_fn().to(device)
    
    loss_val = evaluation_step(
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        device=device,
        save_to=output_dir).avg
    return loss_val



def training_step(train_loader, model, criterion, optimizer, epoch,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  grad_clip=None, print_freq=500):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (hr_imgs, lr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)
        lr_imgs = lr_imgs.reshape(-1,1,512,512)
        hr_imgs = hr_imgs.reshape(-1,1,512,512)
        # Move to default device
        lr_imgs = lr_imgs.to(device,dtype=torch.float)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device,dtype=torch.float)  # (batch_size (N), 3, 96, 96), in [-1, 1]
        
        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored

    
def evaluation_step(dataloader,
                    model,
                    criterion,
                    device,
                    u_water=0.0192,
                    save_to=None):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    if type(save_to)==str:
        os.makedirs(save_to,exist_ok=True)
        hu_transformer = dataset.HUTransformer(u_water=u_water)
    else:
        hu_transformer = lambda x:x
    losses = AverageMeter()
    
    with torch.no_grad():
#     model.train()  # training mode enables batch normalization

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        # Batches
        for i, (hr_imgs, lr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.reshape(-1,1,512,512)
            hr_imgs = hr_imgs.reshape(-1,1,512,512)
            # Move to default device
            lr_imgs = lr_imgs.to(device,dtype=torch.float)  # (batch_size (N), 1, 24, 24), imagenet-normed
            hr_imgs = hr_imgs.to(device,dtype=torch.float)  # (batch_size (N), 1, 96, 96), in [-1, 1]
            
            # Forward prop.
            sr_imgs = model(lr_imgs)  # (N, 1, 96, 96), in [-1, 1]

            # Loss
            loss = criterion(sr_imgs, hr_imgs)  # scalar
            losses.update(loss.item(), lr_imgs.size(0))
            
            if save_to:
                np.save(f"{save_to}/{i}.NDCT.npy",torch.Tensor.cpu(hr_imgs).numpy())
                np.save(f"{save_to}/{i}.QDCT.npy",torch.Tensor.cpu(lr_imgs).numpy())
                np.save(f"{save_to}/{i}.PRED.npy",torch.Tensor.cpu(sr_imgs).numpy())
            
                real_img = hu_transformer._inv_transform_(hr_imgs).to('cpu')
                conv_img = hu_transformer._inv_transform_(sr_imgs).to('cpu')
                input_img = hu_transformer._inv_transform_(lr_imgs).to('cpu')
                for _tag, _img_obj in zip(['NDCT','QDCT', 'PRED'],[real_img,input_img,conv_img]):
                    for j in range(real_img.shape[0]):
                        _curr_img_f = os.path.join(save_to,f"{i}.{j}.{_tag}.png")

                        _curr_img = _img_obj[j].squeeze()
                        plt.imshow(_curr_img, cmap='gray')
                        plt.savefig(_curr_img_f)
            
        del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored

    return losses
