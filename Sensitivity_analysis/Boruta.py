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
        Data_file = 'C:/Users/jm538/Desktop/Code/TV/Data/ML/Custom_data/concat.csv'
        data = pd.read_csv(Data_file,header=0)
        data = data.sample(40000)
        My = data.pop('My')

        hint = np.zeros(len(data.columns))
        sum_importances = np.zeros(len(data.columns))
        print("start")
        print(len(My))
        for i in range(config.iteration):
            start = time.time()
            np.random.seed(i + config.start)
            X_shadow = data.apply(np.random.permutation)
            X_shadow.columns = ['shuffled_' + feat for feat in data.columns]

            X_boruto = pd.concat([data, X_shadow], axis=1)

            mdl = RandomForestRegressor(max_depth=5
                                        )
            mdl.fit(X_boruto, My)
            feature_imp_x = mdl.feature_importances_[:len(data.columns)]
            feature_imp_shuffled = mdl.feature_importances_[len(data.columns):]
            sum_importances += mdl.feature_importances_[:len(data.columns)]
            hint += (feature_imp_x > feature_imp_shuffled.max())
            end = time.time()
            print(f"iteratiom {i}, duration time {start-end :.5f} sec")

        usable_data1 = pd.DataFrame({'hints': hint,'importances':sum_importances}, index = data.columns.to_list())


        with pd.ExcelWriter(os.path.join('Data','Test2.xlsx'), engine="xlsxwriter") as writer:
                    usable_data1.to_excel(writer, sheet_name="variables")
        print('Program End')

    except KeyboardInterrupt:
        print("Canceld by user...")



