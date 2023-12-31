import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

Data_file = 'Data/2023-12-28/Data.csv'
data = pd.read_csv(Data_file,header=0,skiprows=[0,2,3])
data_cleaned = data.loc[:, (data.max(axis=0) != data.min(axis=0)) & ~data.isna().all(axis=0)]
FyFL = data_cleaned.pop('Car.FyFL')
FyFR = data_cleaned.pop('Car.FyFR')
FyRL = data_cleaned.pop('Car.FyRL')
FyRR = data_cleaned.pop('Car.FyRR')

hints1 = np.zeros(len(data_cleaned.columns))
for i in range(200):
    start = time.time()
    np.random.seed(i)
    print('Cal_FyFl', i)
    X_shadow = data_cleaned.apply(np.random.permutation)
    X_shadow.columns = ['shuffled_' + feat for feat in data_cleaned.columns]

    X_boruto = pd.concat([data_cleaned, X_shadow], axis=1)

    mdl = RandomForestRegressor(max_depth=6)
    mdl.fit(X_boruto, FyFL)
    feature_imp_x = mdl.feature_importances_[:len(data_cleaned.columns)]
    feature_imp_shuffled = mdl.feature_importances_[len(data_cleaned.columns):]
    hints1 += (feature_imp_x > feature_imp_shuffled.max())
    end = time.time()
    print(f"iteratiom {i}, duration time {start-end :.5f} sec")

usable_data1 = pd.DataFrame(hints1,index=data_cleaned.columns)

with pd.ExcelWriter('FxFL.xlsx', engine="xlsxwriter") as writer:
            usable_data1.to_excel(writer, sheet_name="variables")
