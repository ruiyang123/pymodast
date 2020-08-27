"""
    In order to use this script, we need to define :

        path_to_dataset_folder : str
        output_path : str
            the path to the folder for storing all optimal points for each datasets
        input_length : int
            the number of input variables
        path_to_observation : str
            path to the observaions(mesures) file
        methode : str
            meta-model (kriging, xgbregressor)
        variables : list
            input variables for the meta-modeling
        ranges : dict
            the ranges (domains) for input variables

    At last, we can generate lots of optimal points saved as (.csv) can be used directly in the script run_simulations.
    And each point correspond one dataset.
"""
from __future__ import print_function
import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import StandardScaler

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

from pymodast.meta_model import kriging, multi_xgb
from pymodast.validate import prepare_data
from pymodast.utils import get_inputs_from_aoc, get_observations_from_aoc
from pymodast.generate_px import prepare_cases_2d


def optimize(dataset_path, aoc_path, lhpt_path, method="kriging", n=None, output_path=None):
    base_folder = os.path.dirname(os.path.dirname(dataset_path))
    variables, ranges = get_inputs_from_aoc(aoc_path)
    if output_path == None:
        output_path = os.path.join(base_folder, "PX_opt_{}".format(method))
    if os.path.exists(output_path) == 0:
        os.mkdir(output_path)
    # prepare data
    X, Y, inputs, outputs = prepare_data(dataset_path)
    # Normalisation
    scaler = StandardScaler()
    scaler.fit(Y)
    Y_norm = scaler.transform(Y)
    # Define the meta-model
    if method == 'kriging':
        model = kriging(X, Y_norm)
    elif method == 'xgboost':
        model = multi_xgb(X, Y_norm)

    # import observations from a excel file
    dict_obs, data_obs = get_observations_from_aoc(aoc_path, lhpt_path)
    observations = data_obs.values[0]
    obs_names = data_obs.columns

    index_filter_pred = []
    for name in obs_names:
        index_filter_pred.append(list(outputs).index(name))
    input_length = len(inputs)
    # define the domain of inputs variables
    max_bound = np.zeros(input_length)
    min_bound = np.zeros(input_length)
    i = 0
    for key in ranges.keys():
        max_bound[i] = ranges[key][1]
        min_bound[i] = ranges[key][0]
        i += 1
    bounds = (min_bound, max_bound)

    # PSO
    def RMSE_loss(x):
        y_pred = scaler.inverse_transform(model.predict(x))
        rmse = np.mean((y_pred[:, index_filter_pred] - observations) ** 2, axis=1)
        rmse = np.sqrt(rmse)
        return rmse

    def MAD_loss(x):
        y_pred = scaler.inverse_transform(model.predict(x))
        mad = np.mean(abs(y_pred[:, index_filter_pred] - observations), axis=1)
        return mad

    def MAX_loss(x):
        y_pred = scaler.inverse_transform(model.predict(x))
        max_loss = np.max(abs(y_pred[:, index_filter_pred] - observations), axis=1)
        return max_loss

    def std_loss(x):
        y_pred = scaler.inverse_transform(model.predict(x))
        std = np.std(y_pred[:, index_filter_pred] - observations, axis=1)
        return std

    def mix_loss(x):
        rmse = RMSE_loss(x)
        max_loss = MAX_loss(x)
        std = std_loss(x)
        a = 0.6
        b = 0.2
        c = 1 - a - b
        return (a * rmse + b * max_loss + c * std)

    # Initialize swarm
    options = {'c1': 1.2, 'c2': 1.2, 'w': 0.7}
    pd_pos = pd.DataFrame(columns=variables)

    if n == None:
        n = input_length
    for i in range(n):
        # Call instance of PSO with bounds argument
        optimizer = ps.single.GlobalBestPSO(n_particles=60, dimensions=input_length, options=options, bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(RMSE_loss, iters=500)

        # plot_cost_history(cost_history=optimizer.cost_history)
        # plt.show()

        # export optimisation position as .csv file
        pos = np.round(pos)
        pos = pd.DataFrame(pos.reshape(1, -1))
        pos.columns = variables
        pd_pos = pd_pos.append(pos)

    pos_name = "PX_opt_method{}_size{}.csv".format(method, np.shape(X)[0])
    pd_pos.to_csv(os.path.join(output_path, pos_name))
    return None



if __name__ == "__main__":
    mode = "2D"

    if mode == "1D":
        aoc_path = "./BV_6LE/BV2016_6LE_Modif.aoc.xml"
        lhpt_path = "./BV_6LE/BV2016_6LE_Modif.lhpt.xml"
        dataset_path ="./BV_6LE/dataset/PX_size190_methodeMinDist.csv"
        optimize(dataset_path, aoc_path, lhpt_path)
    elif mode == "2D":
        aoc_path = "./2D_BV2016_4LE_14Fk/BV2016_4LE_SsCA01.aoc.xml"
        lhpt_path = "./2D_BV2016_4LE_14Fk/BV2016_4LE_SsCA01.lhpt.xml"
        dataset_path ="./2D_BV2016_4LE_14Fk/dataset/PX_size140_methodeMinDist.csv"
        method = "kriging"
        optimize(dataset_path, aoc_path, lhpt_path, method = "kriging")



        px_opt_path = "./2D_BV2016_4LE_14Fk/PX_opt_kriging/PX_opt_methodkriging_size140.csv"
        i2s_path = './2D_BV2016_4LE_14Fk/zones_frottements_BV2016.i2s'
        cas_path = "./2D_BV2016_4LE_14Fk/t2d.cas"
        base_folder = os.path.dirname(os.path.dirname(px_opt_path))
        output_folder = os.path.join(base_folder,"PX_opt_{}/".format(method))
        prepare_cases_2d(px_opt_path, i2s_path, cas_path, output_folder)
