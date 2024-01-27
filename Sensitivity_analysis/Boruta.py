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
        sample_num = 50000
        today= datetime.today()
        Data_file = 'Data/ML/Custom_data/concat_random_torque_vectoring_v.csv'
        data = pd.read_csv(Data_file,header=0)
        sampling_data = data.sample(sample_num,random_state=0)
        My = sampling_data.pop('My')

        hint = np.zeros(len(sampling_data.columns))
        sum_importances = np.zeros(len(sampling_data.columns))
        print("start")
        print(len(My))
        for i in range(config.iteration):
            if i % int((config.iteration))/5 == 0:
                  sampling_data = data.sample(sample_num,random_state=i)
                  My = sampling_data.pop('My')
            start = time.time()
            np.random.seed(i + config.start)
            X_shadow = sampling_data.apply(np.random.permutation)
            X_shadow.columns = ['shuffled_' + feat for feat in sampling_data.columns]

            X_boruto = pd.concat([sampling_data, X_shadow], axis=1)

            mdl = RandomForestRegressor(max_depth=5
                                        )
            mdl.fit(X_boruto, My)
            feature_imp_x = mdl.feature_importances_[:len(sampling_data.columns)]
            feature_imp_shuffled = mdl.feature_importances_[len(sampling_data.columns):]
            sum_importances += mdl.feature_importances_[:len(sampling_data.columns)]
            hint += (feature_imp_x > feature_imp_shuffled.max())
            end = time.time()
            print(f"iteratiom {i}, duration time {start-end :.5f} sec")

        usable_data1 = pd.DataFrame({'hints': hint,'importances':sum_importances}, index = sampling_data.columns.to_list())


        with pd.ExcelWriter(os.path.join('Data','check_variables_random_torque_vectoring_v.xlsx'), engine="xlsxwriter") as writer:
                    usable_data1.to_excel(writer, sheet_name="variables")
        print('Program End')

    except KeyboardInterrupt:
        print("Canceld by user...")



