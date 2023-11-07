# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.optim
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------------------------------------------------------
# define the process of loading model state and predict new data
def model_load_predict(net, pred_data_dir, pred_result_dir):
    # loading the model parameters
    net.load_state_dict(torch.load('0-model.pt'))
    feat_max = torch.load('2-feat-max.pth')
    feat_min = torch.load('2-feat-min.pth')
    label_max = torch.load('2-label-max.pth')
    label_min = torch.load('2-label-min.pth')
    # load data from predicting file
    pred_data_numpy = np.loadtxt(pred_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
    pred_data_tensor_o = torch.from_numpy(pred_data_numpy)
    pred_data_tensor = (pred_data_tensor_o - feat_min) / (feat_max - feat_min)
    # forward process
    if pred_data_tensor.ndim == 1:
        pred_result = net.forward(torch.unsqueeze(pred_data_tensor, dim=1))
        pred_result = pred_result * (label_max - label_min) + label_min
        pred_tensor = torch.cat([torch.unsqueeze(pred_data_tensor_o, dim=1), pred_result], 1)
    else:
        pred_result = net.forward(pred_data_tensor)
        pred_result = pred_result * (label_max - label_min) + label_min
        pred_tensor = torch.cat([pred_data_tensor_o, pred_result], 1)
    np.savetxt(pred_result_dir, pred_tensor.detach().numpy(), fmt='%.6f', delimiter=',')


# ----------------------------------------------------------------------------------------------------------------------
# define the process of loading model state and predict new data
def random_para_predict(net, pred_result_dir, mean_var, std_var, coeff_var, mcs_times, dis_type):
    # loading the model parameters
    net.load_state_dict(torch.load('0-model.pt'))
    feat_max = torch.load('2-feat-max.pth')
    feat_min = torch.load('2-feat-min.pth')
    label_max = torch.load('2-label-max.pth')
    label_min = torch.load('2-label-min.pth')
    # generate random parameters
    if dis_type == 2:  # log normal distribution
        temp_mean = np.array(mean_var)
        temp_std = np.array(std_var)
        mean_var_log = np.log(temp_mean * temp_mean / np.sqrt(temp_std * temp_std + temp_mean * temp_mean))
        std_var_log = np.sqrt(np.log(temp_std * temp_std / temp_mean / temp_mean + 1))
        mean_var_1 = (np.full((mcs_times, 2), mean_var_log)).T
        std_var_1 = (np.full((mcs_times, 2), std_var_log)).T
    elif dis_type == 1:  # normal distribution
        temp_mean = np.array(mean_var)
        temp_std = np.array(std_var)
        mean_var_1 = (np.full((mcs_times, 2), temp_mean)).T
        std_var_1 = (np.full((mcs_times, 2), temp_std)).T

    rand_num = np.random.normal(0, 1, (2, mcs_times))
    mat_upper = sp.linalg.cholesky(np.array([[1.0, coeff_var], [coeff_var, 1.0]]))
    pred_data_numpy = np.dot(mat_upper, rand_num)

    if dis_type == 2:
        pred_data_numpy = np.exp(mean_var_1 + std_var_1 * pred_data_numpy)
    elif dis_type == 1:
        pred_data_numpy = mean_var_1 + std_var_1 * pred_data_numpy

    pred_data_tensor_o = torch.from_numpy(pred_data_numpy.T)
    pred_data_tensor = (pred_data_tensor_o - feat_min) / (feat_max - feat_min)
    # forward process
    if pred_data_tensor.ndim == 1:
        pred_result = net.forward(torch.unsqueeze(pred_data_tensor.to(torch.float32), dim=1))
        pred_result = pred_result * (label_max - label_min) + label_min
        pred_tensor = torch.cat([torch.unsqueeze(pred_data_tensor_o, dim=1), pred_result], 1)
    else:
        pred_result = net.forward(pred_data_tensor.to(torch.float32))
        pred_result = pred_result * (label_max - label_min) + label_min
        pred_tensor = torch.cat([pred_data_tensor_o, pred_result], 1)
    np.savetxt(pred_result_dir, pred_tensor.detach().numpy(), fmt='%.6f', delimiter=',')


# ----------------------------------------------------------------------------------------------------------------------

