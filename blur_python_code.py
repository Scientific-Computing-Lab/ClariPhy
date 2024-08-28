
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv
from skimage.metrics import structural_similarity as ssim

from PSNR import PSNR
import subprocess

# from skimage import rescale


n = 178
m = 87
avg_time = 40
nt = np.zeros((2,71))
imgs_PSNR = np.zeros((2,71))
imgs_SSIM = np.zeros((2,71))
d = 3

base_save_dir = fr'./restormer/blur/blury_imgs'
base_imgs_path = rf'./RaylePhotos/gravity_-600_amplitode_0.1_atwood_0.1'



def gaussian_convolution(a, it):
    icen = np.ceil(d * erfinv(0.99995))
    jcen = icen
    aa = np.abs(np.arange(-icen, icen+1, 1))
    bb = np.dot(np.reshape(aa,(-1, 1)), np.ones((1, len(aa))))
    gg1 = bb**2
    aa = abs(np.arange(-jcen, jcen+1, 1))
    bb = np.ones((len(aa), 1)) * aa
    gg2 = bb**2
    gg = 1 / d**2 / np.pi*np.exp(-(gg1 + gg2) / d**2)
    # 'sum(sum(gg))=', sum(sum(gg))
    # ahar = convolve2d(a, gg, mode='same')
    # nt[it + 1, d] = len(np.where(ahar[1:-1 - 10, 1: -1 - 10] < 250 & ahar[1: -1 - 10, 1: -1 - 10] > 5));

    # icen = np.ceil(d * erfinv(0.99995))
    # jcen = icen
    # aa = np.abs(np.arange(icen, -icen - 1, -1))
    # bb = np.outer(aa, np.ones_like(aa))
    # gg1 = bb ** 2
    # aa = np.abs(np.arange(jcen, -jcen - 1, -1))
    # bb = np.outer(np.ones_like(aa), aa)
    # gg2 = bb ** 2
    # gg = 1 / (d ** 2 * np.pi) * np.exp(-(gg1 + gg2) / d ** 2)
    # print('sum(sum(gg))=', np.sum(np.sum(gg)))
    # ahar = convolve2d(a, gg, mode='same')
    # nt[it + 1, d] = np.sum((ahar[:-10, :-10] < 250) & (ahar[:-10, :-10] > 5))
    return ahar, nt

def rgb_to_gray(img):
    # grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    Avg = Avg
    # grayImage = img.copy()
    #
    # for i in range(3):
    #     grayImage[:, :, i] = Avg

    return Avg#grayImage

def cut_to_size(multiple_of, img):
    img = img[0:img.shape[0] - img.shape[0]%multiple_of, 0:img.shape[1] - img.shape[1]%multiple_of]
    global n
    n = img.shape[0] - img.shape[0]%multiple_of
    global m
    m = img.shape[1] - img.shape[1]%multiple_of

    return img

def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)

def spatial_blur(img, it, d):
    s = np.zeros((d + 1, d + 1))
    ft = np.zeros((n, m))
    for i in range(d + 1):
        for j in range(d + 1):
            if ((i + 0.5) ** 2 + (j + 0.5) ** 2 < d ** 2):
             a = 1
            elif ((i - 0.5) ** 2 + (j - 0.5) ** 2 > d ** 2):
             a = 0
            elif ((i + 0.5) ** 2 + (j - 0.5) ** 2 < d ** 2):
             yi05 = np.sqrt(d ** 2 - (i + 0.5) ** 2)
             if ((i - 0.5) ** 2 + (j + 0.5) ** 2 < d ** 2):
                 xj05 = np.sqrt(d ** 2 - (j + 0.5) ** 2)
                 a = 1 - abs(polygon_area(np.array([i + 0.5, j + 0.5, i + 0.5]), np.array([ yi05, xj05, j + 0.5])))
             else:
                 yim05 = np.sqrt(d ** 2 - (i - 0.5) ** 2)
                 a = abs(polygon_area(np.array([i - 0.5, j - 0.5, i + 0.5, j - 0.5]),np.array([ i + 0.5, yi05, i - 0.5, yim05])))
            else:
             xjm05 = np.sqrt(d ** 2 - (j - 0.5) ** 2)
             if ((i - 0.5) ** 2 + (j + 0.5) ** 2 < d ** 2):
                 xjm05 = np.sqrt(d ** 2 - (j + 0.5) ** 2)
                 a = abs(polygon_area(np.array([i - 0.5, j - 0.5, xjm05, j - 0.5]), np.array([ xj05, j + 0.5, i - 0.5, j + 0.5])))
             else:
                 yim05 = np.sqrt(d ** 2 - (i - 0.5) ** 2)
                 a = abs(polygon_area(np.array([i - 0.5, j - 0.5, i - 0.5]), np.array([yim05, xjm05, j - 0.5])))
            s[i, j] = a
    for i in range(n):
        for j in range(m):
             for ii in range(max(0, i - d), min(n, i + d)):
                 for jj in range(max(0, j - d), min(m, j + d)):
                     ft[i, j] += s[abs(ii - i), abs(jj - j) ]

    ct = str(it)
    if 0 < it < 10:
        ct = '0' + ct
    if it % 10 == 0:
        ct = ct[0]
    ahar = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for ii in range(max(0, i - d), min(n, i + d)):
                for jj in range(max(0, j - d), min(m, j + d)):
                    # print([i, j, ii, jj])
                    ahar[ii, jj] += img[i, j] * s[abs(ii - i), abs(jj - j)] / ft[ii, jj]
    # nt[it, d] = len(np.flatnonzero((ahar < 0.9961) & (ahar > 0.0039)))
    ahar[ahar>1.0] = 1.0
    ahar[ahar < 0.0]= 0.0
    return ahar, nt

def temp_blur(imgs):
    blur_imgs = np.mean(imgs, 0)
    return blur_imgs


def read_orig_img(img_path):
    cur_img = plt.imread(img_path)

    cur_img = rgb_to_gray(cur_img)
    cur_img = cur_img[10:10 + n, 106:106 + m]

    clr1 = 0.2532353
    clr2 = 0.18475294
    cur_img[cur_img  != clr1] = clr2

    cur_img[cur_img == clr1] = 1.0
    cur_img[cur_img == clr2] = 0.0

    cur_img_cut = cut_to_size(8, cur_img)

    # cur_img_cut = np.resize(cur_img, (176, 80))

    return cur_img, cur_img_cut


def deblur(img_dir, result_dir, img_name, model_name='motion_deblurring'):
    p = subprocess.Popen(('./to_deblur.sh {} {} {}'.format(img_dir, result_dir, model_name)), shell=True)
    p.wait()
    img = plt.imread(os.path.join(result_dir, f'Motion_Deblurring/{img_name}'))
    img = rgb_to_gray(img)
    return img


def create_figs(imgs, time, base_save_dir):#orig_img, spatial_blury_img, temp_blury_img, spatial_temp_blury_img, deblur_img

    imgs_names = ['orig_img', 'spatial_blury_img', 'temp_blury_img', 'spatial_temp_blury_img', 'deblur_img']

    fig1, ax = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Bluring')
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)

    for k, ax in enumerate(ax.flatten()):
        save_imgs_path = os.path.join(base_save_dir, f'blur_proc_{time}')
        if not os.path.exists(save_imgs_path):
            os.mkdir(save_imgs_path)
        plt.imsave(os.path.join(save_imgs_path, f'{imgs_names[k]}_time_0.{time}_d_{d}.png'), imgs[k], cmap='gray')
        ax.imshow(imgs[k], cmap='gray')
        ax.set_title(f'{imgs_names[k]}: t = 0.{time}')
        ax2.plot(imgs[k][:, 50], label=imgs_names[k], marker='.')
        ax3.plot(imgs[k][75:100, 50], label=imgs_names[k], marker='.')

    # print(f'spatial blur range: {[np.min(imgs[1]), np.max(imgs[1])]}')
    # print(f'orig range:{[np.min(imgs[0]), np.max(imgs[0])]}')
    ax2.plot(imgs[4][:,50], label=imgs_names[4], marker='.')
    ax2.legend()
    ax3.plot(imgs[4][75:100, 50], label=imgs_names[4], marker='.')
    ax3.legend()
    plt.imsave(os.path.join(save_imgs_path, f'{imgs_names[4]}_time_0.{time}_d_{d}.png'), imgs[4], cmap='gray')
    fig1.savefig(os.path.join(save_imgs_path, fr'bluriness_proc_time_0.{time}_d_{d}.png'))
    fig2.savefig(os.path.join(save_imgs_path, fr'example_cut_time_0.{time}_d_{d}.png'))
    fig3.savefig(os.path.join(save_imgs_path, fr'example_cut_time_zoom_0.{time}_d_{d}.png'))



def tests(sp_tmp_blur_img, deblur_img, orig_img, col):
    imgs_PSNR[0, col] = PSNR(orig_img, sp_tmp_blur_img)
    imgs_SSIM[0, col] = ssim(orig_img, sp_tmp_blur_img)
    nt[0, col] = len(np.flatnonzero((sp_tmp_blur_img < 0.9961) & (sp_tmp_blur_img > 0.0039)))

    imgs_PSNR[1, col] = PSNR(orig_img, deblur_img)
    imgs_SSIM[1, col] = ssim(orig_img, deblur_img)
    nt[1, col] = len(np.flatnonzero((deblur_img < 0.9961) & (deblur_img > 0.0039)))


def main():

    orig_imgs = []
    blury_imgs_spatial = []
    blury_imgs_temporal = []
    blury_imgs_spatial_temporal = []
    deblur_imgs = []


    for time in np.arange(1, 75):
        if time%10 == 0:
            time = int(time/10)
            time = str(time)
        else:
            time = str(time)
            time = time.zfill(2)

        img_path = os.path.join(base_imgs_path, f'time=0.{time}.png')
        img_save_path = os.path.join(base_save_dir, f'original_imgs_time_0.{time}.png')

        cur_img, cur_img_cut = read_orig_img(img_path)
        cur_blury_img, nt = spatial_blur(cur_img_cut, int(time), d)

        blury_imgs_spatial.append(cur_blury_img)
        orig_imgs.append(cur_img_cut)

    for time in np.arange(3, 73):

        time_str = str(time)
        time_str = time_str.zfill(2)
        cur_temp_blur = temp_blur(orig_imgs[time-2:time+3])
        blury_imgs_temporal.append(cur_temp_blur)
        cur_spatial_temp_blur = temp_blur(blury_imgs_spatial[time-2:time+3])

        blur_path = os.path.join(base_save_dir, 'spatial_temporal_blur')

        if not os.path.exists(blur_path):
            os.mkdir(blur_path)
        plt.imsave(os.path.join(blur_path, f'blury_imgs_time_0.{time_str}_d_{d}.png'), cur_spatial_temp_blur, cmap='gray')

        blury_imgs_spatial_temporal.append(cur_spatial_temp_blur)
        deblur_path = os.path.join(base_save_dir, 'spatial_temporal_deblur')
        if not os.path.exists(deblur_path):
            os.mkdir(deblur_path)

        cur_deblur_img = deblur(os.path.join(blur_path, f'blury_imgs_time_0.{time_str}_d_{d}.png'),deblur_path, f'blury_imgs_time_0.{time_str}_d_{d}.png')
        # plt.imsave(os.path.join(deblur_path, f'deblury_imgs_time_0.{time_str}_d_{d}.png'), cur_deblur_img, cmap='gray')
        deblur_imgs.append(cur_deblur_img)

        tests(cur_spatial_temp_blur, cur_deblur_img, orig_imgs[time], time-2)

        if time % 10 == 0:
            create_figs([orig_imgs[time-1], blury_imgs_spatial[time-1], blury_imgs_temporal[time-3],
                         blury_imgs_spatial_temporal[time-3], deblur_imgs[time-3]], time, base_save_dir)

    fig1, ax1 = plt.subplots(1, 1)
    plt.suptitle('nt')
    ax1.plot(np.arange(0.03, 0.73, 0.01), nt.transpose()[1:])
    ax1.legend(['blur', 'deblur'])
    ax1.set_xlabel('time')
    ax1.set_ylabel('nt')
    fig1.savefig(os.path.join(base_save_dir, fr'nt.png'))

    fig2, ax2 = plt.subplots(1, 1)
    plt.suptitle('PSNR')
    ax2.plot(np.arange(0.03, 0.73, 0.01),imgs_PSNR.transpose()[1:])
    ax2.legend(['blur', 'deblur'])
    ax2.set_xlabel('time')
    ax2.set_ylabel('PSNR')
    fig2.savefig(os.path.join(base_save_dir, fr'PSNR.png'))

    fig3, ax3 = plt.subplots(1, 1)
    plt.suptitle('SSIM')
    ax3.plot(np.arange(0.03, 0.73, 0.01),imgs_SSIM.transpose()[1:])
    ax3.legend(['blur', 'deblur'])
    ax3.set_xlabel('time')
    ax3.set_ylabel('SSIM')
    fig3.savefig(os.path.join(base_save_dir, fr'SSIM.png'))

    plt.show()

    print(nt, imgs_PSNR, imgs_SSIM )


if __name__ == "__main__":
    main()
