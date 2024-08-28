import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv
from skimage.metrics import structural_similarity as ssim

from PSNR import PSNR
import subprocess
import shutil
import json

from blur_python_code import deblur



def try_specific_model(save_path,img_new_name, model_name='motion_deblurring'):
    deblur(os.path.join(save_path, 'blurred.png'), save_path, 'blurred.png', model_name=model_name)
    shutil.copy(os.path.join(save_path, 'Motion_Deblurring', 'blurred.png'),
              os.path.join(save_path,img_new_name))# 'motion_deblurring.png'))


def test_results(orig_img, new_img):
    PSNRvalue = PSNR(orig_img[:,:,0:3], new_img)
    SSIMvalue = ssim(orig_img[:,:,0:3], new_img, channel_axis=2)

    return PSNRvalue, SSIMvalue


#TODO: add whole folder


original_path = r'./restormer/blur/Data_Base_1/gravity_-600_amplitode_0.1_atwood_0.02/original'
blured_path = r'./restormer/blur/Data_Base_1/gravity_-600_amplitode_0.1_atwood_0.02/blur_d_5_t_9'

models_path = r'./restormer/Restormer/Motion_Deblurring/pretrained_models'


base_path = r'./restormer/blur'
all_SSIM =[]
all_PSNR = []
time = np.array([])
all_imgs = os.listdir(blured_path)
for file in all_imgs: #[0::4]

    f = blured_path.split('/')
    img_name = fr'{f[9]}_{f[10]}'#_time_0.{f[11].split(".")[1]}'
    fol_name = file.split('.png')[0]
    time = np.append(time, float(fol_name.split('_')[1]))
    save_path = os.path.join(base_path, 'test_single', img_name, fol_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cut_fig, cut_ax = plt.subplots(1, 1)
    plt.suptitle('Pixel Value')
    cut_fig_zi, cut_ax_zi = plt.subplots(1, 1)
    plt.suptitle('Pixel Value')


    original_img = plt.imread(os.path.join(original_path, file))
    blured_img = plt.imread(os.path.join(blured_path, file))

    cut_ax.plot(original_img[:, 50, 0], label='original img', marker='.')
    cut_ax.plot(blured_img[:, 50, 0], label='blurred img', marker='.')

    cut_ax_zi.plot(original_img[75:125, 50, 0], label='original img', marker='.')
    cut_ax_zi.plot(blured_img[75:125, 50, 0], label='blurred img', marker='.')

    cut_ax.legend(loc = 'upper right')
    cut_ax_zi.legend(loc = 'upper right')

    cut_ax.set_xlabel('row')
    cut_ax.set_ylabel('value')

    cut_ax_zi.set_xlabel('row')
    cut_ax_zi.set_ylabel('value')

    cut_fig.savefig(os.path.join(save_path, fr'example_cut_original_blur.png'))
    cut_fig_zi.savefig(os.path.join(save_path, fr'example_cut_original_blur_zoom.png'))



    plt.imsave(os.path.join(save_path, 'original.png'), original_img)
    plt.imsave(os.path.join(save_path, 'blurred.png'), blured_img)

    model_names = os.listdir(models_path)
    model_names = [i.split('.')[0] for i in model_names if i.endswith('.pth')]
    PSNRs = []
    SSIMs = []
    for model_name in model_names:
        try_specific_model(save_path, f'{model_name}.png', model_name=model_name)
        deblurred_img = plt.imread(os.path.join(save_path, f'{model_name}.png'))

        psnr_val, ssim_val = test_results(original_img, deblurred_img)
        PSNRs.append(psnr_val)
        SSIMs.append(ssim_val)
        cut_ax.plot(deblurred_img[:, 50, 0], label=f'deblurred - {model_name}', marker='.')
        cut_ax_zi.plot(deblurred_img[75:125, 50, 0], label=f'deblurred - {model_name}', marker='.')
    cut_ax.get_legend().remove()
    cut_ax_zi.get_legend().remove()

    cut_ax_zi.legend(loc = 'upper right')
    cut_ax.legend(loc = 'upper right')
    cut_ax.set_xlabel('row')
    cut_ax.set_ylabel('value')

    cut_ax_zi.set_xlabel('row')
    cut_ax_zi.set_ylabel('value')
    cut_fig.savefig(os.path.join(save_path, fr'example_cut_original_all.png'))
    cut_fig_zi.savefig(os.path.join(save_path, fr'example_cut_original_all_zoom.png'))
    plt.close(cut_fig)
    plt.close(cut_fig_zi)
    # import pdb; pdb.set_trace()
    psnr_val, ssim_val = test_results(original_img, blured_img[:,:,0:3])
    PSNRs.append(psnr_val)
    SSIMs.append(ssim_val)

    print(SSIMs, PSNRs)
    all_SSIM.append(SSIMs)
    all_PSNR.append(PSNRs)

fig1, ax1 = plt.subplots(1,1)
plt.suptitle('SSIM')
fig2, ax2 = plt.subplots(1,1)
plt.suptitle('PSNR')
all_SSIM = np.array(all_SSIM)
all_PSNR = np.array(all_PSNR)
model_names.append('blurred')
for i in range(len(model_names)):
    ax1.plot(time, all_SSIM[:, i], label=model_names[i])
    ax2.plot(time, all_PSNR[:, i], label=model_names[i])
ax1.legend()
ax2.legend()
ax2.set_xlabel('time')
ax1.set_xlabel('time')
ax1.set_ylabel('SSIM')
ax2.set_ylabel('PSNR')

pd.Da

fig1.savefig(os.path.join(base_path, 'test_single', img_name, 'SSIM.png'))
fig2.savefig(os.path.join(base_path, 'test_single', img_name, 'PSNR.png'))

# with open("ssim", "w") as fp:
#     json.dump(all_SSIM, fp)
#
# with open("psnr", "w") as fp:
#     json.dump(all_PSNR, fp)

# >>> with open("test", "r") as fp:
# ...     b = json.load(fp)

# import pdb; pdb.set_trace()
print(all_SSIM)
print(all_PSNR)





# deblur(os.path.join(save_path, 'blurred.png'), save_path, 'blurred.png', model_name='motion_deblurring')
# os.rename(os.path.join(save_path,'Motion_Deblurring', 'blurred.png'), os.path.join(save_path,'Motion_Deblurring', 'motion_deblurring.png'))
# deblur(os.path.join(save_path, 'blurred.png'), save_path, 'blurred.png', model_name='net_g_latest')
# os.rename(os.path.join(save_path,'Motion_Deblurring', 'blurred.png'), os.path.join(save_path, 'Motion_Deblurring', 'net_g_latest.png'))
