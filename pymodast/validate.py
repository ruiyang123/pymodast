"""
    In order to use this script, we need to define :

        path_to_dataset_folder
        path_to_plot : str
            the path to the folder for storing all plots. in this script, we can generate some validation plots from the datasets.
        number_crue : int
            the number of crue
        input_length : int
            the number of input variables
        k : int
            the number of folders for k-folde cross-validation
        method : str
            meta-model (kriging, xgbregressor)

    At last, we can generate lots of plots. And each plot correspond one dataset. We generate also a file named <val_dict.npy> which stocks all results(RMSE,MSD,MAD) for all datasets.
"""
from crue10.utils.settings import NCSIZE
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import seaborn as sns

import time
from tqdm import *
import utils
import meta_model


def get_crue_names(dataset_path):
    '''
    :param path_to_dataset: str
        the path to dataset(.csv)
    :return: list
        a list contains the name of crue
    '''
    dataset = pd.read_csv(dataset_path)
    input_length = utils.get_input_length(dataset)
    num_crue = utils.get_num_crue(dataset)
    col_names = dataset.columns
    inputs = col_names[0:input_length]
    outputs = col_names[input_length:]
    num_sections = int(len(outputs) / num_crue)
    crue_names = []
    for i in range(num_crue):
        crue_names.append(outputs[i * num_sections].split('/')[0])
    return crue_names


def prepare_data(dataset_path):
    ''' prepare data to the meta-modeling

    :param path_to_dataset: str
        the path to dataset(.csv)
    :return:
        X : numpy array, shape(n_samples, n_features)
        Y : numpy array, shape(n_samples, n_outputs)
        inputs : list, [name of input variables]
        outputs : list, [name of output variables]
    '''
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.replace(0, np.nan)
    dataset = dataset.dropna(axis=0)
    input_length = utils.get_input_length(dataset)

    col_names = dataset.columns
    inputs = col_names[0:input_length]
    outputs = col_names[input_length:]
    output_length = len(outputs)

    data_ori = pd.DataFrame(dataset)
    data = data_ori.values

    X = data[:, 0:input_length]
    Y = data[:, input_length:]
    return X, Y, inputs, outputs


def k_folder(X, Y, number_crue, k, method="kriging"):
    '''

    :param X: numpy array, shape(n_samples,n_features)
        inputs
    :param Y: numpy array, shape(n_samples, n_outputs)
        outputs
    :param number_crue: int
        the number of crue
    :param k: int
        the number of folders for k-folde cross-validation
    :param methode: str
        meta-model (kriging, xgbregressor)
    :return:
        list_RMSE : numpy array, shape(n_samples, n_crues)
            Root Mean Square Error
        list_MSD : numpy array, shape(n_samples, n_crues)
            Mean Signed Deviation
        list_MAD : numpy array, shape(n_samples, n_crues)
            Mean Absolute Error
        total_rmse : float
        total_msd : float
        total_mad : float

    '''
    scaler = StandardScaler()
    scaler.fit(Y)
    Y_norm = scaler.transform(Y)
    num_targets = np.shape(Y)[1]
    num_samples = np.shape(Y)[0]
    kf = KFold(n_splits=k)

    list_RMSE = np.zeros((np.shape(Y)[0], number_crue))
    list_MSD = np.zeros((np.shape(Y)[0], number_crue))
    list_MAD = np.zeros((np.shape(Y)[0], number_crue))
    total_rmse = 0
    total_msd = 0
    total_mad = 0

    index1 = 0
    for train_index, test_index in tqdm(kf.split(X)):
        time.sleep(0.1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_norm[train_index], Y[test_index]

        if method == "kriging":
            model = meta_model.kriging(X_train, y_train)
        elif method == "xgboost":
            model = meta_model.multi_xgb(X_train, y_train)
        else:
            print("wrong name for <methode>")
            return 0
        y_pred = scaler.inverse_transform(model.predict(X_test))

        num_subsample = np.shape(y_pred)[0]
        for index, rmse in enumerate((y_pred - y_test) ** 2):
            rmse = rmse.reshape(number_crue, -1)
            rmse = np.mean(rmse, axis=1)
            number = index1 * num_subsample + index
            if number <= num_samples:
                list_RMSE[number, :] = np.sqrt(rmse)
                total_rmse = total_rmse + np.sum(rmse) * (num_targets / number_crue)
        for index, msd in enumerate((y_pred - y_test)):
            msd = msd.reshape(number_crue, -1)
            msd = np.mean(msd, axis=1)
            number = index1 * num_subsample + index
            if number <= num_samples:
                list_MSD[number, :] = msd
                total_msd += np.sum(msd) * (num_targets / number_crue)

        for index, mad in enumerate(abs(y_pred - y_test)):
            mad = mad.reshape(number_crue, -1)
            mad = np.mean(mad, axis=1)
            number = index1 * num_subsample + index
            if number <= num_samples:
                list_MAD[number, :] = mad
                total_mad += np.sum(mad) * (num_targets / number_crue)
        index1 += 1
    total_rmse = np.sqrt(total_rmse / (num_samples * num_targets))
    return list_RMSE, list_MSD, list_MAD, total_rmse, total_msd / (num_samples * num_targets), total_mad / (
            num_samples * num_targets)


def val_plot(RMSE, MAD, MSD, total_rmse, total_mad, total_msd, crue_names, output_path, plot_name):
    # Plot histogramme for each crue and each criterion
    fig = plt.figure(figsize=(15, 8))
    sns.set(font_scale=2.5)
    index = 0
    number_crue = len(crue_names)
    for i in range(3):
        plt.subplot(1, 3, index + 1)
        index += 1
        for j in range(number_crue):
            if i == 0:
                ax1 = sns.kdeplot(RMSE[:, j], label=crue_names[j], lw=5)
                ax1.legend(fontsize=17)
                plt.xlabel("RMSE")
            elif i == 1:
                ax2 = sns.kdeplot(MAD[:, j], label=crue_names[j], lw=5)
                ax2.legend(fontsize=17)
                plt.xlabel("MAD")
            else:
                ax3 = sns.kdeplot(MSD[:, j], label=crue_names[j], lw=5)
                ax3.legend(fontsize=17)
                plt.xlabel("MSD")
    plt.suptitle("total_RMSE : {} \t total_MAD : {} \t total_MSD : {} ".format(format(total_rmse, '.3g'),
                                                                               format(total_mad, '.3g'),
                                                                               format(total_msd, '.3g')))
    plt.show()
    # save plot
    fig.savefig(os.path.join(output_path, plot_name))


def validate(dataset_path, output_path, k_f, method="kriging", plot_name=None):
    crue_names = get_crue_names(dataset_path)
    number_crue = len(crue_names)
    X, Y, inputs, outputs = prepare_data(dataset_path)
    sample_size = np.shape(Y)[0]
    RMSE, MSD, MAD, total_rmse, total_msd, total_mad = k_folder(X, Y, number_crue, k_f, method=method)
    if not plot_name:
        plot_name = "val_plot_method{}_size{}.png".format(method, sample_size)
    val_plot(RMSE, MAD, MSD, total_rmse, total_mad, total_msd, crue_names, output_path, plot_name)
    return None


if __name__ == "__main__":
    dataset_path = "./CE/dataset/PX_size120_methodeMinDist.csv"
    output_path = "./CE/validation_plots_kriging"
    k_f = 10
    validate(dataset_path, output_path, k_f)