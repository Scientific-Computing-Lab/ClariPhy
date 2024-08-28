#!/bin/bash
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate pytorch181
cd ./restormer/Restormer/
python demo.py --task Motion_Deblurring --input_dir $1 --result_dir $2 --model_name $3
