# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import os
import torch
from torch.nn import init
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as plt_fm
from Neural_Network_Class_Define import Neural_Network_Class
from Neural_Network_Train_Test import training_testing_process
from Neural_Network_Train_Test import training_testing_incremental
from Neural_Network_Predict import model_load_predict
from Neural_Network_Predict import random_para_predict

# ----------------------------------------------------------------------------------------------------------------------
# main program
# define the running type: 2--train and test; 3--load model state and conduct incremental train;
# 4--load model state and predict new data; 5--automatically generate the random parameters and perform the prediction.
run_type = 2
# ----------------------------------------------------------------------------------------------------------------------
# define the file path
# training data file path
if os.sep == "/":  # linux platform
    train_data_dir = r'./train-data/train-data.csv'
else:  # windows platform
    train_data_dir = r'.\\train-data\\train-data.csv'
# testing data file path
if os.sep == "/":  # linux platform
    test_data_dir = r'./test-data/test-data.csv'
else:  # windows platform
    test_data_dir = r'.\\test-data\\test-data.csv'
# predicting data file path
if os.sep == "/":  # linux platform
    pred_data_dir = r'./predict/data.csv'
else:  # windows platform
    pred_data_dir = r'.\\predict\\data.csv'
# predicting result file path
if os.sep == "/":  # linux platform
    pred_result_dir = r'./predict/result.csv'
else:  # windows platform
    pred_result_dir = r'.\\predict\\result.csv'
# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
#    (input_size, hidden_size)            (hidden_size, hidden_size)             (hidden_size, output_size)
# ----------------------------------------------------------------------------------------------------------------------
# define the sizes of input vector, hidden vector and output vector
input_size = 2
hidden_size = 9
output_size = 1
slice_num = input_size
# define the number of features
# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
# the type of activation function: 1--sigmoid function; 2--ReLU function; 3--tanh function
actifun_type1 = 2
actifun_type2 = 2
# ----------------------------------------------------------------------------------------------------------------------
# the type of optimizer: 1--SGD; 2--Adam; 3--RMSprop; 4--Adagrad
optim_type = 3
# the type of loss function: 1--mean squared error function; 2--cross entropy error function; 3--PoissonNLLLoss function
loss_type = 1
# ----------------------------------------------------------------------------------------------------------------------
# define the batch size
batch_size = 6
# define the learning rate
learn_r = 0.002
# define the weight_decay
weight_d = 0.001
# define the training cycle
train_loop = 100
# ----------------------------------------------------------------------------------------------------------------------
# define the mean, standard deviation and correlation coefficient for run-type == 4
mean_var = [20, 10]
std_var = [4, 2]
coeff_var = -0.3
mcs_times = 10000
dis_type = 2  # 1--normal distribution; 2--log normal distribution
# ----------------------------------------------------------------------------------------------------------------------
# setting font
font_tnr_reg = plt_fm.FontProperties('Times New Roman', size=14, stretch=0)
# ----------------------------------------------------------------------------------------------------------------------
# set the model instance
Net_Model = Neural_Network_Class(input_size, hidden_size, output_size, actifun_type1, actifun_type2)
print(Net_Model)
# ----------------------------------------------------------------------------------------------------------------------
# define training and testing function of Net_Model
# testing the model with test data
if run_type == 2:
    # initializing the weight parameters
    for layer in Net_Model.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)

    loss_h1, loss_h2, simu_score_train, simu_score_test, feat_max, feat_min, label_max, label_min = \
        training_testing_process(Net_Model, learn_r, weight_d, batch_size, train_loop,
                                 optim_type, loss_type,
                                 train_data_dir, test_data_dir,
                                 slice_num, output_size)
    # plot the relationship between loss_value and iteration step
    fig = plt.figure()
    loss_df_1 = pd.DataFrame(loss_h1, columns=['step', 'loss'])
    loss_df_2 = pd.DataFrame(loss_h2, columns=['step', 'loss'])
    plt.plot(loss_df_1['loss'].values, 'go', markersize=2)
    plt.plot(loss_df_2['loss'].values, 'mo', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Loss', fontproperties=font_tnr_reg)
    plt.show()

    # save training data in the log file
    score_train_df = pd.DataFrame(simu_score_train, columns=['step', 'score-per', 'score-r2'])
    score_test_df = pd.DataFrame(simu_score_test, columns=['step', 'score-per', 'score-r2'])
    loss_df_1.to_csv('1-loss-process_train.txt', sep='\t', index=False)
    loss_df_2.to_csv('1-loss-process_test.txt', sep='\t', index=False)
    score_train_df.to_csv('1-score-train-process.txt', sep='\t', index=False)
    score_test_df.to_csv('1-score-test-process.txt', sep='\t', index=False)
    torch.save(feat_max, '2-feat-max.pth')
    torch.save(feat_min, '2-feat-min.pth')
    torch.save(label_max, '2-label-max.pth')
    torch.save(label_min, '2-label-min.pth')
# ----------------------------------------------------------------------------------------------------------------------
# define the module of loading model state and conducting the incremental training
# testing the model with test data
if run_type == 3:
    # load the model state and the boundaries of features and labels: max, min.
    Net_Model.load_state_dict(torch.load('0-model.pt'))
    feat_max = torch.load('2-feat-max.pth')
    feat_min = torch.load('2-feat-min.pth')
    label_max = torch.load('2-label-max.pth')
    label_min = torch.load('2-label-min.pth')
    # incremental train process
    loss_h1, loss_h2, simu_score_train, simu_score_test = \
        training_testing_incremental(Net_Model, learn_r, weight_d, batch_size, train_loop,
                                     optim_type, loss_type,
                                     train_data_dir, test_data_dir,
                                     slice_num, feat_max, feat_min, label_max, label_min, output_size)
    # plot the relationship between loss_value and iteration step
    fig = plt.figure()
    loss_df_1 = pd.DataFrame(loss_h1, columns=['step', 'loss'])
    loss_df_2 = pd.DataFrame(loss_h2, columns=['step', 'loss'])
    plt.plot(loss_df_1['loss'].values, 'go', markersize=2)
    plt.plot(loss_df_2['loss'].values, 'mo', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Loss', fontproperties=font_tnr_reg)
    plt.show()

    # save training data in the log file
    score_train_df = pd.DataFrame(simu_score_train, columns=['step', 'score-per', 'score-r2'])
    score_test_df = pd.DataFrame(simu_score_test, columns=['step', 'score-per', 'score-r2'])
    loss_df_1.to_csv('1-loss-process_train.txt', sep='\t', index=False)
    loss_df_2.to_csv('1-loss-process_test.txt', sep='\t', index=False)
    score_train_df.to_csv('1-score-train-process.txt', sep='\t', index=False)
    score_test_df.to_csv('1-score-test-process.txt', sep='\t', index=False)
    torch.save(feat_max, '2-feat-max.pth')
    torch.save(feat_min, '2-feat-min.pth')
    torch.save(label_max, '2-label-max.pth')
    torch.save(label_min, '2-label-min.pth')
# ----------------------------------------------------------------------------------------------------------------------
# define state loading and prediction function of Net_Model
if run_type == 4:
    model_load_predict(Net_Model, pred_data_dir, pred_result_dir)
# ----------------------------------------------------------------------------------------------------------------------
# define the random parameter generation module and Monte-Carlo prediction
if run_type == 5:
    if dis_type == 1 or dis_type == 2:
        random_para_predict(Net_Model, pred_result_dir, mean_var, std_var, coeff_var, mcs_times, dis_type)
# ----------------------------------------------------------------------------------------------------------------------
