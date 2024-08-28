import os
import numpy as np
import matplotlib.pyplot as plt
import shutil


DB_base_path = r'./restormer/blur/Data_Base_1'
BD_org_path = r'./restormer/blur/Data_Base_org_small_1'


folder_stride = 2
permutation_stride = 2
time_stride = 2

train_size = 0.8
test_size = 0.2

if not os.path.exists(BD_org_path):
    os.makedirs(BD_org_path)

folder_names = os.listdir(DB_base_path)
folder_names = folder_names[0::folder_stride]
np.random.shuffle(folder_names)


test_names = folder_names[0:int(np.floor(len(folder_names)*test_size))]
train_names = folder_names[int(np.floor(len(folder_names)*test_size)):]

train_path = os.path.join(BD_org_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
sz = len(train_names)
print(sz)
for name in train_names:
    print(f'train: {name}')
    old_path = os.path.join(DB_base_path, name)
    target_folder = os.path.join(old_path, 'original')
    input_folders = [folder for folder in os.listdir(old_path) if folder != 'original']
    input_folders = input_folders[0::permutation_stride]
    for folder in input_folders[1:]:
        time_list = os.listdir(os.path.join(old_path, folder))
        time_list = time_list[0::time_stride]
        for img in time_list:
            #input
            img_org_path = os.path.join(os.path.join(old_path, folder), img)
            img_new_name = f'{name}_{folder}_{img}'
            img_new_path = os.path.join(os.path.join(train_path, 'input'), img_new_name)
            if not os.path.exists(os.path.join(train_path,  'input')):
                os.makedirs(os.path.join(train_path, 'input'))
            shutil.copy(img_org_path, img_new_path)

            #target
            img_org_path = os.path.join(os.path.join(old_path, 'original'), img)
            img_new_name = f'{name}_{folder}_{img}'
            img_new_path = os.path.join(os.path.join(train_path, 'target'), img_new_name)
            if not os.path.exists(os.path.join(train_path, 'target')):
                os.makedirs(os.path.join(train_path, 'target'))
            shutil.copy(img_org_path, img_new_path)

test_path = os.path.join(BD_org_path, 'test')
if not os.path.exists(test_path):
    os.makedirs(test_path)
sz = len(test_names)
for name in test_names:
    print(f'test: {name}')
    old_path = os.path.join(DB_base_path, name)
    target_folder = os.path.join(old_path, 'original')
    input_folders = [folder for folder in os.listdir(old_path) if folder != 'original']
    input_folders = input_folders[0::permutation_stride]
    for folder in input_folders[1:]:
        time_list = os.listdir(os.path.join(old_path, folder))
        time_list = time_list[0::time_stride]
        for img in time_list:
            #input
            img_org_path = os.path.join(os.path.join(old_path, folder), img)
            img_new_name = f'{name}_{folder}_{img}'
            img_new_path = os.path.join(os.path.join(test_path, 'input'), img_new_name)
            if not os.path.exists(os.path.join(test_path,  'input')):
                os.makedirs(os.path.join(test_path, 'input'))
            shutil.copy(img_org_path, img_new_path)

            #target
            img_org_path = os.path.join(os.path.join(old_path, 'original'), img)
            img_new_name = f'{name}_{folder}_{img}'
            img_new_path = os.path.join(os.path.join(test_path, 'target'), img_new_name)
            if not os.path.exists(os.path.join(test_path, 'target')):
                os.makedirs(os.path.join(test_path, 'target'))
            shutil.copy(img_org_path, img_new_path)


