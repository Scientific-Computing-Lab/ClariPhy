from math import log10, sqrt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = plt.imread(r"./restormer/blur/blury_imgs/cut_blury_imgs_time_0.40_d_3.png")
    new = plt.imread(r"./restormer/blur/blury_imgs/test_restormer/Motion_Deblurring/cut_blury_imgs_time_0.40_d_3.png")
    original = original[:, :, 0:3]
    PSNRvalue = PSNR(original, new)
    SSIMvalue = ssim(original, new, channel_axis=2)
    print(f"PSNR value is {PSNRvalue} dB")
    print(f"SSIMvalue is {SSIMvalue} ")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(original)
    ax[0].set_title('original')
    ax[1].imshow(new)
    ax[1].set_title('new')
    plt.suptitle(f'PNSR: {PSNRvalue}, SSIM: {SSIMvalue}')
    plt.show()



if __name__ == "__main__":
    main()
