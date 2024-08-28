import os
import numpy as np
import matplotlib.pyplot as plt

from blur_python_code import read_orig_img, polygon_area, temp_blur
from multiprocessing import Process


def create_s_mat(d, n = 176, m = 80):
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
                    a = 1 - abs(polygon_area(np.array([i + 0.5, j + 0.5, i + 0.5]), np.array([yi05, xj05, j + 0.5])))
                else:
                    yim05 = np.sqrt(d ** 2 - (i - 0.5) ** 2)
                    a = abs(polygon_area(np.array([i - 0.5, j - 0.5, i + 0.5, j - 0.5]),
                                         np.array([i + 0.5, yi05, i - 0.5, yim05])))
            else:
                xjm05 = np.sqrt(d ** 2 - (j - 0.5) ** 2)
                if ((i - 0.5) ** 2 + (j + 0.5) ** 2 < d ** 2):
                    xjm05 = np.sqrt(d ** 2 - (j + 0.5) ** 2)
                    a = abs(polygon_area(np.array([i - 0.5, j - 0.5, xjm05, j - 0.5]),
                                         np.array([xj05, j + 0.5, i - 0.5, j + 0.5])))
                else:
                    yim05 = np.sqrt(d ** 2 - (i - 0.5) ** 2)
                    a = abs(polygon_area(np.array([i - 0.5, j - 0.5, i - 0.5]), np.array([yim05, xjm05, j - 0.5])))
            s[i, j] = a
    for i in range(n):
        for j in range(m):
            for ii in range(max(0, i - d), min(n, i + d)):
                for jj in range(max(0, j - d), min(m, j + d)):
                    ft[i, j] += s[abs(ii - i), abs(jj - j)]



    return s, ft


def spatial_blur_with_s(img, it, s, d, ft, n=176, m=80):

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
    ahar[ahar > 1.0] = 1.0
    ahar[ahar < 0.0] = 0.0
    return ahar


def create_DB_signle_folder(folder, base_save_dir, start, stop):

    save_dir = os.path.join(base_save_dir, os.path.basename(folder))
    save_path_orig = os.path.join(save_dir, 'original')
    if not os.path.exists(save_path_orig):
        os.makedirs(save_path_orig)

    orig_imgs = []
    blury_imgs_spatial_temporal = []
    for time_1 in np.arange(start, stop):

        if time_1 % 10 == 0:
            time = int(time_1 / 10)
            time = str(time)
        else:
            time = str(time_1)
            time = time.zfill(2)

        img_path = os.path.join(folder, f'time=0.{time}.png')
        cur_img, cur_img_cut = read_orig_img(img_path)

        # if int(time) < 10:
        #     time = int(time)*10
        #     time = str(time)
        #     print(time)
        time_1 = str(time_1)
        plt.imsave(os.path.join(save_path_orig, f'time_0.{time_1.zfill(2)}.png'), cur_img_cut, cmap='gray')
        orig_imgs.append(cur_img_cut)

    for t in range(3, 15, 2):
        for d in range(3, 10, 2):
            print(f'{os.path.basename(folder)}: t: {t}, d: {d}')
            s, ft = create_s_mat(d)
            cur_save_path = os.path.join(save_dir, f'blur_d_{d}_t_{t}')
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            blury_imgs_spatial = []
            for time, i in zip(np.arange(start, stop),  np.arange(start-1, stop-1)):
                cur_blury_img = spatial_blur_with_s(orig_imgs[i], int(time), s, d, ft)
                blury_imgs_spatial.append(cur_blury_img)

            for cur_t in range(start+(t-2), stop-(t-2)):
                cur_sp_temp_blur = temp_blur(blury_imgs_spatial[cur_t-t//2:cur_t+t//2+1])
                blury_imgs_spatial_temporal.append(cur_sp_temp_blur)
                cur_t = str(cur_t)
                plt.imsave(os.path.join(cur_save_path, f'time_0.{cur_t.zfill(2)}.png'), cur_sp_temp_blur, cmap='gray')


def main():
    base_save_dir = fr'./restormer/blur/Data_Base_1'
    base_data_dir = rf'./RaylePhotos'

    start_time = 1
    stop_time = 72
    names = os.listdir(base_data_dir)
    folder_names = [os.path.join(base_data_dir, folder) for folder in names if 'gravity' in folder]
    procs = []


    for folder in folder_names[100:]: #0:80, now 80:100 #amp01_at0.48 need to do again
        print(folder)
        proc = Process(target=create_DB_signle_folder, args=(folder, base_save_dir, start_time, stop_time))
        procs.append(proc)
        proc.start()
        print('proc started')

    # complete the processes
    for proc in procs:
        proc.join()
        print('proc ended')


if __name__ == "__main__":
    main()
