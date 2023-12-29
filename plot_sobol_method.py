import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
## import csv_file

Path = './2023-12-21'
SI_data= 'Test_data_si.xlsx'
Data_file = os.path.join(Path, SI_data)

data = pd.read_excel(Data_file,sheet_name='Si',header=0,index_col=0)

s1_row = data.loc['S1']

# Plotting
s1_row.plot(kind='bar', figsize=(15, 5))  # Using a bar plot for better visibility
plt.title('S1 Values Across Different Columns')
plt.xlabel('Columns')
plt.ylabel('S1 Values')
plt.show()