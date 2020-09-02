from __future__ import print_function
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re

from pymodast.meta_model import kriging, multi_xgb
from pymodast.validate import prepare_data
from pymodast.utils import get_observations_from_aoc


def plot_opt(dataset_path, dataset_opt_path, aoc_path, lhpt_path, method="kriging", output_path=None):
    base_folder = os.path.dirname(os.path.dirname(dataset_path))
    output_path = os.path.join(base_folder, "opt_plots_{}".format(method))
    if os.path.exists(output_path) == 0:
        os.mkdir(output_path)

    opt_dict = {}

    df_opt = pd.read_csv(dataset_opt_path)

    # prepare data
    X, Y, inputs, outputs = validate.prepare_data(dataset_path)
    scaler = StandardScaler()
    scaler.fit(Y)
    Y_norm = scaler.transform(Y)
    # create the meta-model
    if method == 'kriging':
        model = meta_model.kriging(X, Y_norm)
    elif method == 'xgboost':
        model = meta_model.multi_xgb(X, Y_norm)
    input_length = len(inputs)
    for index, row in df_opt.iterrows():
        plot_name = '{}.png'.format(index)
        rmse_name = '{}.xlsx'.format(index)

        opt = row
        df = pd.DataFrame(columns=['Crue', 'Sect', 'Type', 'Z'])
        outputs = df_opt.columns[input_length:]
        in_value = opt.values[0:input_length]
        out_value = opt.values[input_length:]
        Y_pred = scaler.inverse_transform(model.predict(in_value.reshape(1, -1)))

        for i, name in enumerate(outputs):
            crue = name.split('/')[0]
            Sect = name.split('/')[1]
            Type = "simulation"
            value = out_value[i]
            new_df = pd.DataFrame([[crue, Sect, Type, value]], columns=['Crue', 'Sect', 'Type', 'Z'])
            df = df.append(new_df)
            value = Y_pred[0][i]
            Type = "prediction"
            new_df = pd.DataFrame([[crue, Sect, Type, value]], columns=['Crue', 'Sect', 'Type', 'Z'])
            df = df.append(new_df)

        obs, obs_df = utils.get_observations_from_aoc(aoc_path, lhpt_path)
        outputs = obs_df.columns
        out_value = obs_df.values
        for i, name in enumerate(outputs):
            crue = name.split('/')[0]
            Sect = name.split('/')[1]
            Type = "observation"
            value = float(out_value[0][i])
            new_df = pd.DataFrame([[crue, Sect, Type, value]], columns=['Crue', 'Sect', 'Type', 'Z'])
            df = df.append(new_df)

        # get sect_id
        pattern = re.compile(r'\d+.\d*')
        dict_sec = {}
        list_pos = []
        for sec in df['Sect'].unique():
            pos = float(re.search(pattern, sec).group())
            list_pos.append(pos)
            dict_sec[sec] = pos
        sorted_pos = np.sort(list_pos)
        for key in dict_sec.keys():
            index = list(sorted_pos).index(dict_sec[key])
            dict_sec[key] = index

        df['sect_id'] = df.apply(lambda row: dict_sec[row.Sect], axis=1)

        fig = plt.figure(figsize=(15, 8))
        sns.set(font_scale=2.35)
        ax = sns.scatterplot(x="sect_id", y="Z", hue="Crue", style="Type", s=170, data=df)
        ax.legend(markerscale=4)

        # Calculate RMSE
        rmse_df = pd.DataFrame(columns=["crue", "rmse"])

        total_rmse = np.array([])

        crue_names = df["Crue"].unique()
        for crue_name in crue_names:
            s1 = list(df[(df['Crue'] == crue_name) & (df['Type'] == 'simulation')]["sect_id"].values)
            s2 = list(df[(df['Crue'] == crue_name) & (df['Type'] == 'observation')]["sect_id"].values)
            simul = df[(df['Crue'] == crue_name) & (df['Type'] == 'simulation')]["Z"].values
            obser = df[(df['Crue'] == crue_name) & (df['Type'] == 'observation')]["Z"].values
            index_filter_pred = []
            for i in s2:
                index_filter_pred.append(s1.index(i))
            simul = simul[index_filter_pred]
            total_rmse = np.concatenate((total_rmse, simul - obser), axis=None)
            rmse = np.sqrt(np.mean((simul - obser) ** 2))
            new_df = pd.DataFrame([[crue_name, rmse]], columns=['crue', 'rmse'])
            rmse_df = rmse_df.append(new_df)
            print(crue_name, rmse)

        total_rmse = np.sqrt(np.mean(total_rmse ** 2))
        new_df = pd.DataFrame([['total rmse', total_rmse]], columns=['crue', 'rmse'])
        rmse_df = rmse_df.append(new_df)
        rmse_df.to_excel(os.path.join(output_path, rmse_name), index=False)
        plt.suptitle("total_RMSE : {}".format(format(total_rmse, '.3g')))
        fig.savefig(os.path.join(output_path, plot_name))
        
    best_columns = []
    for input_name in inputs :
        best_columns.append(input_name)
    best_columns.append("rmse")


    best_df = pd.DataFrame([np.append(best_fk,best_rmse)],columns=best_columns)
    best_df.to_excel(os.path.join(output_path, "best_config.xlsx"), index=False)


if __name__ == "__main__":
    dataset_path = "./2D_BV2016_4LE_14Fk/dataset/PX_size140_methodeMinDist.csv"
    dataset_opt_path = "./2D_BV2016_4LE_14Fk/opt_dataset_kriging_rep/kriging_opt_rep_dataset.csv"
    aoc_path = "./2D_BV2016_4LE_14Fk/BV2016_4LE_SsCA01.aoc.xml"
    lhpt_path = "./2D_BV2016_4LE_14Fk/BV2016_4LE_SsCA01.lhpt.xml"
    plot_opt(dataset_path, dataset_opt_path, aoc_path, lhpt_path, method="kriging", output_path=None)
