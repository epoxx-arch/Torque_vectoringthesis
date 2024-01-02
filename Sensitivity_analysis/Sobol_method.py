from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import os
import pandas as pd

# Parameters
Path = './2023-12-21'
file_name = 'Test_data_2.csv'
SI_data= 'Test_data_2_si.xlsx'
Data_file = os.path.join(Path, file_name)

# import csv file 

data = pd.read_csv(Data_file,header=0,skiprows=[0,2,3])
data_cleaned = data.loc[:, (data.max(axis=0) != data.min(axis=0)) & ~data.isna().all(axis=0)]
print(data_cleaned)
del data


# Extracting specific columns and dropping them
FyFL = data_cleaned.pop('Car.FyFL')
FyFR = data_cleaned.pop('Car.FyFR')
FyRL = data_cleaned.pop('Car.FyRL')
FyRR = data_cleaned.pop('Car.FyRR')


# Sobol's method preparation
# Find boundary of the value


min_max_values = [[data_cleaned[col].min(), data_cleaned[col].max()] for col in data_cleaned.columns]

names =data_cleaned.columns.to_list()


problem ={
    'num_vars' : len(data_cleaned.columns),
    'names' : names,
    'bounds' : min_max_values

}

n_samples = 4096

param_values = saltelli.sample(problem, n_samples)


# Dummy model evaluation function
# Replace this with your actual model function
def model_evaluation(params):
    # Example: Sum of parameters (replace with your model logic)
    return np.sum(params, axis=0)


# Run the model for each generated sample
model_outputs = np.array([model_evaluation(sample) for sample in param_values])

# Perform Sobol analysis
Si = sobol.analyze(problem, model_outputs)

# Print the first order, total order, and second order sensitivities

df = pd.DataFrame.from_dict(data=Si, orient='index',columns=names)

with pd.ExcelWriter(os.path.join(Path, SI_data), engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Si")

