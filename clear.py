import os
import glob
from parser import args
from tqdm import tqdm

folder_path = args.test_path
file_pattern = "*_res.png" 

file_list = glob.glob(os.path.join(folder_path, file_pattern))

print(">>>>>> Starting Clear\n")

# 删除预测的结果
for file_path in tqdm(file_list):
    os.remove(file_path)

print("\n<<<<<< Clear finished.\n")