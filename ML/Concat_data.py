import os
import pandas as pd

path = 'C:/Users/User/Desktop/Code/TV/Data/2024_01_27/random_torque_vectoring'
output_path = 'Data/ML/Custom_data/new_concat.csv'

file_list = os.listdir(path)

input_path = os.path.join(path , file_list.pop(0))
df = pd.read_csv(input_path, delimiter=',',low_memory=False)
num = len(file_list)

for i,file in enumerate(file_list):
    temp_df =pd.read_csv(os.path.join(path,file), delimiter=',',low_memory=False)
    df = pd.concat([df,temp_df], ignore_index=True)
    print("%d/%d" %(i,num))

df.to_csv(output_path, index=False)
    