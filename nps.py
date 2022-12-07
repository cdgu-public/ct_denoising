#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# def get_nps_2d_img(img_arr):
#     fft_shifted = np.abs(np.fft.fft2(img_arr))
#     fft_shifted = np.fft.fftshift(fft_shifted)
#     return fft_shifted


def get_nps_2d_img(img_arr,bx=1.,by=1.,fft_n=None):
    if not fft_n:
        fft_n = np.rint(np.sqrt(np.prod(img_arr.shape))).astype(int)
    fft_shifted = (np.abs(np.fft.fft2(img_arr-img_arr.mean(), s=(fft_n, fft_n))) /
                np.sqrt(np.prod(img_arr.shape))) ** 2.
    fft_shifted = fft_shifted*bx*by
    fft_shifted = np.fft.fftshift(fft_shifted)
    return fft_shifted


def fft2DRadProf(ft, d=1.):
    coord = np.fft.fftshift(np.fft.fftfreq(ft.shape[0], d))
    image = np.fft.fftshift(ft)

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    center = np.squeeze(np.array([x[0, coord == 0.0], y[coord == 0.0, 0]]))
    r = np.hypot(x - center[0], y - center[1])
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = np.rint(r_sorted).astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    cl = int(np.rint(np.sum(coord >= 0)))
    return coord[coord >= 0], radial_prof[:cl]


def massCenter(a):
    arr = np.array(a, dtype=np.float)
    totalMass = arr.sum()
    sh = arr.shape
    if totalMass == 0:
        return (0,) * len(sh)
    if len(arr.shape) == 3:
        xcm = np.sum(np.sum(np.sum(arr, axis=2),
                            axis=1) * (np.indices([sh[0]])[0, :]))
        ycm = np.sum(np.sum(np.sum(arr, axis=2),
                            axis=0) * (np.indices([sh[1]])[0, :]))
        zcm = np.sum(np.sum(np.sum(arr, axis=1),
                            axis=0) * (np.indices([sh[2]])[0, :]))
        return xcm / totalMass, ycm / totalMass, zcm / totalMass
    elif len(arr.shape) == 2:
        xcm = np.sum(np.sum(arr, axis=1) * (np.indices([sh[0]])[0, :]))
        ycm = np.sum(np.sum(arr, axis=0) * (np.indices([sh[1]])[0, :]))
        return xcm / totalMass, ycm / totalMass
    elif len(arr.shape) == 1:
        xcm = np.sum(arr * np.indices([sh[0]]))
        return xcm / totalMass
    else:
        return None
    
    
def get_radial_nps(array,
                   dxdy=1,
                   angle_sample=400,
                   kernel_l=5,
                   rad=25,
                   bdia_n=10,
                   fft_n=None,
                   cm=None,
                  ):
    # radii = [50. / dxdy, 25. / dxdy]
    # angles = [np.linspace(0, 2*np.pi * (1.-1./20.), 20)]
    # angles += [np.linspace(0, 2*np.pi * (1.-1./8.), 8)]
    radii = [rad / dxdy]
    angles = [np.linspace(0, 2*np.pi * (1.-1./angle_sample), angle_sample)]
    # angles += [np.linspace(0, 2*np.pi * (1.-1./8.), 8)]
    # bdia = int(np.rint(10. / dxdy))
    bdia = int(np.rint(bdia_n / dxdy))
    coords = [(0, 0)]

    # subroi coords
    for ind, r in enumerate(radii):
        for ang in angles[ind]:
            x, y = np.rint((r * np.sin(ang), r * np.cos(ang)))
            coords.append((x, y))
    if not fft_n:
        fft_n = np.rint(np.sqrt(np.prod(array.shape))).astype(int)
    kernel = np.ones(kernel_l) / kernel_l
    f, b = np.ones(kernel_l), np.ones(kernel_l)
    lineCollection = []
    
#     cm = (x_0+fov_size/2,x_0+fov_size/2)
    if not cm:
        cm = np.rint(massCenter(array > -300))
#     cm = (fov_size/2,fov_size/2)
    lines = []
    fboxes = []
    xs = []
    ys = []
    for c in coords:
        box = np.copy(array[int(cm[0]+c[0]-bdia): int(cm[0]+c[0]+bdia),
                            int(cm[1]+c[1]-bdia): int(cm[1]+c[1]+bdia)])
        fbox = (np.abs(np.fft.fft2(box-box.mean(), s=(fft_n, fft_n))) /
                np.sqrt(np.prod(box.shape))) ** 2.
        fboxes.append(fbox)
        x, y = fft2DRadProf(fbox, dxdy)
        xs.append(x)
        ys.append(y)
        lines.append(y)
        lineCollection.append(y)
    ps = np.array(lines).mean(0)
    yc = np.convolve(np.hstack((f * y[0], ps, b * y[-1])), kernel,
                     mode='same')[kernel_l: -kernel_l]
#     fs = x * 10.
    fs = x * 2.
    psAll = np.array(lineCollection).mean(0)
    ycAll = np.convolve(np.hstack((f * y[0], psAll, b * y[-1])), kernel,
                        mode='same')[kernel_l: -kernel_l]
    
    # plotting
    
    nps_2d_img = get_nps_2d_img(array)
    rad_nps_data = {
        'x':fs,
        'y':yc
        }
    
    return nps_2d_img, rad_nps_data, (psAll, ycAll)


def save_img(img_arr,output='output.png'):
    if output.endswith('npy'):
        with open(output,'wb') as f:
            np.save(img_arr,output)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_arr, cmap='gray')
        plt.savefig(output)

        
def parse_npy(file):
    with open(file,'rb') as f:
        arr = np.load(f)
    return arr


def main(ndct_img,
         qdct_img,
         pred_img,
         output='./output',
         x_0=220,y_0=128,fov_size=128,bx=1.,by=1.,
         rad=20,
         bdia_n=10,
         fft_n=None,
        ):
    output = os.path.abspath(output)
    os.makedirs(output,exist_ok=True)
    
    noise = ndct_img-qdct_img
    pred_noise = pred_img-qdct_img
    ndct_fov = ndct_img[x_0:x_0+fov_size,y_0:y_0+fov_size]
    qdct_fov = qdct_img[x_0:x_0+fov_size,y_0:y_0+fov_size]
    pred_fov = pred_img[x_0:x_0+fov_size,y_0:y_0+fov_size]
    noise_fov = ndct_fov-qdct_fov
    pred_noise_fov = pred_fov-qdct_fov
    
    save_img(ndct_img,os.path.join(output,'ndct_img.png'))
    save_img(qdct_img,os.path.join(output,'qdct_img.png'))
    save_img(pred_img,os.path.join(output,'pred_img.png'))
    
    save_img(noise,os.path.join(output,'noise.png'))
    save_img(pred_noise,os.path.join(output,'pred_noise.png'))
    
    save_img(ndct_fov,os.path.join(output,'ndct_fov.png'))
    save_img(qdct_fov,os.path.join(output,'qdct_fov.png'))
    save_img(noise_fov,os.path.join(output,'noise_fov.png'))
    save_img(pred_noise_fov,os.path.join(output,'pred_noise_fov.png'))
    
    nps_2d = get_nps_2d_img(noise,fft_n=fft_n)
    nps_2d_fov = get_nps_2d_img(noise_fov,fft_n=fft_n)
    save_img(nps_2d,os.path.join(output,'ndct_qdct.nps_2d.png'))
    save_img(nps_2d_fov,os.path.join(output,'ndct_qdct_fov.nps_2d.png'))
    
    pred_nps_2d = get_nps_2d_img(pred_noise,fft_n=fft_n)
    pred_nps_2d_fov = get_nps_2d_img(pred_noise_fov,fft_n=fft_n)
    save_img(pred_nps_2d,os.path.join(output,'pred_qdct.nps_2d.png'))
    save_img(pred_nps_2d_fov,os.path.join(output,'pred_qdct_fov.nps_2d.png'))
    
    ndct_fov_nps_2d = get_nps_2d_img(ndct_fov,bx=1.,by=1.,fft_n=fft_n)
    qdct_fov_nps_2d = get_nps_2d_img(qdct_fov,bx=1.,by=1.,fft_n=fft_n)
    pred_fov_nps_2d = get_nps_2d_img(pred_fov,bx=1.,by=1.,fft_n=fft_n)
    save_img(ndct_fov_nps_2d,os.path.join(output,'ndct_fov.nps_2d.png'))
    save_img(qdct_fov_nps_2d,os.path.join(output,'qdct_fov.nps_2d.png'))
    save_img(pred_fov_nps_2d,os.path.join(output,'pred_fov.nps_2d.png'))
    
    results = dict(
        ndct_result_1d = get_radial_nps(ndct_img,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],
        qdct_result_1d = get_radial_nps(qdct_img,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],
        pred_result_1d = get_radial_nps(pred_img,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],

        ndct_fov_result_1d = get_radial_nps(ndct_fov,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],
        qdct_fov_result_1d = get_radial_nps(qdct_fov,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],
        pred_fov_result_1d = get_radial_nps(pred_fov,rad=rad,bdia_n=bdia_n,fft_n=fft_n)[1],
    )
    
    with open(os.path.join(output,'combined.pkl'),'wb') as f:
        pickle.dump(results,f)
    
    
    