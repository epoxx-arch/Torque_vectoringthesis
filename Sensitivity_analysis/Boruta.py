import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime
import os
import argparse

if __name__ == "__main__":
    try:
        ## parser setting
        parser =argparse.ArgumentParser(description="parameter setting")
        parser.add_argument('--output', type = str, default = None)
        parser.add_argument('--iteration', type= int, default = 50)
        parser.add_argument('--start', type= int, default = 0)
        config = parser.parse_args()

        today= datetime.today()
        Data_file = os.path.join('../Data',str(today.date()),'Data.csv')
        data = pd.read_csv(Data_file,header=0,skiprows=[0,2,3])
        data_cleaned = data.loc[:, (data.max(axis=0) != data.min(axis=0)) & ~data.isna().all(axis=0)]

        output  = argparse.ArgumentParser(description='')
        y = data_cleaned.pop(config.output)
        hint = np.zeros(len(data_cleaned.columns))
        sum_importances = np.zeros(len(data_cleaned.columns))
        for i in range(config.iteration):
            start = time.time()
            np.random.seed(i + config.start)
            print('Cal_FyFR', i)
            X_shadow = data_cleaned.apply(np.random.permutation)
            X_shadow.columns = ['shuffled_' + feat for feat in data_cleaned.columns]

            X_boruto = pd.concat([data_cleaned, X_shadow], axis=1)

            mdl = RandomForestRegressor(max_depth=5)
            mdl.fit(X_boruto, y)
            feature_imp_x = mdl.feature_importances_[:len(data_cleaned.columns)]
            feature_imp_shuffled = mdl.feature_importances_[len(data_cleaned.columns):]
            sum_importances += mdl.feature_importances_[:len(data_cleaned.columns)]
            hint += (feature_imp_x > feature_imp_shuffled.max())
            end = time.time()
            print(f"iteratiom {i}, duration time {start-end :.5f} sec")

        usable_data1 = pd.DataFrame({'hints': hint,'importances':sum_importances}, index = data_cleaned.columns.to_list())

        with pd.ExcelWriter(os.path.join('Data',config.output + str(config.start) + '.xlsx'), engine="xlsxwriter") as writer:
                    usable_data1.to_excel(writer, sheet_name="variables")
        print('Program End')

    except KeyboardInterrupt:
        print("Canceld by user...")



