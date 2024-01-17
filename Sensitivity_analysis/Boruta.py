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
        parser.add_argument('--iteration', type= int, default = 10)
        parser.add_argument('--start', type= int, default = 0)
        config = parser.parse_args()

        today= datetime.today()
        Data_file = 'C:/Users/User/Desktop/Code/TV/Data/2024-01-09/Data.csv'
        data = pd.read_csv(Data_file,header=0)

        y1 = data.pop('Car.FyFR')
        y2 = data.pop('Car.FyFL')
        y3 = data.pop('Car.FyRL')
        y4 = data.pop('Car.FyRR')

        y = y1 + y2 + y3 + y4

        hint = np.zeros(len(data.columns))
        sum_importances = np.zeros(len(data.columns))
        for i in range(config.iteration):
            start = time.time()
            np.random.seed(i + config.start)
            print('Cal_FyFR', i)
            X_shadow = data.apply(np.random.permutation)
            X_shadow.columns = ['shuffled_' + feat for feat in data.columns]

            X_boruto = pd.concat([data, X_shadow], axis=1)

            mdl = RandomForestRegressor(max_depth=6)
            mdl.fit(X_boruto, y)
            feature_imp_x = mdl.feature_importances_[:len(data.columns)]
            feature_imp_shuffled = mdl.feature_importances_[len(data.columns):]
            sum_importances += mdl.feature_importances_[:len(data.columns)]
            hint += (feature_imp_x > feature_imp_shuffled.max())
            end = time.time()
            print(f"iteratiom {i}, duration time {start-end :.5f} sec")

        usable_data1 = pd.DataFrame({'hints': hint,'importances':sum_importances}, index = data.columns.to_list())


        with pd.ExcelWriter(os.path.join('Data','Test.xlsx'), engine="xlsxwriter") as writer:
                    usable_data1.to_excel(writer, sheet_name="variables")
        print('Program End')

    except KeyboardInterrupt:
        print("Canceld by user...")



