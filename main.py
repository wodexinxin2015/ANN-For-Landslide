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

# ----------------------------------------------------------------------------------------------------------------------
# main program
# define the running type： 1--train; 2--train and test; 3--load model state and predict new data
# 4--automatically generate the random parameters and perform the prediction.
run_type = 4
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
# define the number of data in training file and testing file
train_data_num = 25
test_data_num = 5
slice_num = 2  # define the number of features
# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
#    (input_size, hidden_size)            (hidden_size, hidden_size)             (hidden_size, output_size)
# ----------------------------------------------------------------------------------------------------------------------
# define the sizes of input vector, hidden vector and output vector
input_size = slice_num
hidden_size = 10
output_size = 1
# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
# the type of activation function: 1--sigmoid function; 2--ReLU function; 3--tanh function
actifun_type1 = 2
actifun_type2 = 1
# ----------------------------------------------------------------------------------------------------------------------
# the type of optimizer: 1--SGD; 2--Adam; 3--RMSprop; 4--Adagrad
optim_type = 3
# the type of loss function: 1--mean squared error function; 2--cross entropy error function; 3--PoissonNLLLoss function
loss_type = 1
# ----------------------------------------------------------------------------------------------------------------------
# define the batch size
batch_size = 4
# define the learning rate
learn_r = 0.001
# define the training cycle
train_loop = 1000
# ----------------------------------------------------------------------------------------------------------------------
# define the mean, standard deviation and correlation coefficient for run-type == 4
mean_var = [20, 10]
std_var = [6, 3]
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
# define training function of Net_Model
# train the model with training data
if run_type == 1:
    # initializing the weight parameters
    for layer in Net_Model.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)

    loss_holder = Net_Model.training_process(learn_r, batch_size, train_loop, optim_type, loss_type, train_data_dir,
                                             slice_num)
    # plot the relationship between loss_value and iteration step
    fig = plt.figure()
    loss_df = pd.DataFrame(loss_holder, columns=['step', 'loss'])
    plt.plot(loss_df['loss'].values, 'ro', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Loss', fontproperties=font_tnr_reg)
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# define training and testing function of Net_Model
# testing the model with test data
if run_type == 2:
    # initializing the weight parameters
    for layer in Net_Model.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)

    loss_holder, simu_score_train, simu_score_test, feat_max, feat_min, label_max, label_min = \
        Net_Model.training_testing_process(learn_r, batch_size, train_loop,
                                           optim_type, loss_type,
                                           train_data_num, test_data_num,
                                           train_data_dir, test_data_dir,
                                           slice_num)
    # plot the relationship between loss_value and iteration step
    fig = plt.figure()
    loss_df = pd.DataFrame(loss_holder, columns=['step', 'loss'])
    plt.plot(loss_df['loss'].values, 'ro', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Loss', fontproperties=font_tnr_reg)
    plt.show()
    # plot the relationship between loss_value and iteration step
    score_train_df = pd.DataFrame(simu_score_train, columns=['step', 'score-per', 'score-cos'])
    score_test_df = pd.DataFrame(simu_score_test, columns=['step', 'score-per', 'score-cos'])

    fig2 = plt.figure()
    plt.plot(score_train_df['score-per'].values, 'g*', markersize=2)
    plt.plot(score_test_df['score-per'].values, 'm*', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Pearson Coefficient', fontproperties=font_tnr_reg)
    plt.show()

    fig3 = plt.figure()
    plt.plot(score_train_df['score-cos'].values, 'g*', markersize=2)
    plt.plot(score_test_df['score-cos'].values, 'm*', markersize=2)
    plt.xticks(fontproperties=font_tnr_reg)
    plt.yticks(fontproperties=font_tnr_reg)
    plt.xlabel('Iteration step', fontproperties=font_tnr_reg)
    plt.ylabel('Cosine similarity', fontproperties=font_tnr_reg)
    plt.show()

    # save training data in the log file
    loss_df.to_csv('1-loss-process.txt', sep='\t', index=False)
    score_train_df.to_csv('1-score-train-process.txt', sep='\t', index=False)
    score_test_df.to_csv('1-score-test-process.txt', sep='\t', index=False)
    torch.save(feat_max, '2-feat-max.pth')
    torch.save(feat_min, '2-feat-min.pth')
    torch.save(label_max, '2-label-max.pth')
    torch.save(label_min, '2-label-min.pth')
# ----------------------------------------------------------------------------------------------------------------------
# define state loading and prediction function of Net_Model
if run_type == 3:
    Net_Model.model_load_predict(pred_data_dir, pred_result_dir)
# ----------------------------------------------------------------------------------------------------------------------
# define the random parameter generation module and Monte-Carlo prediction
if run_type == 4:
    if dis_type == 1 or dis_type == 2:
        Net_Model.random_para_predict(pred_result_dir, mean_var, std_var, coeff_var, mcs_times, dis_type)
# ----------------------------------------------------------------------------------------------------------------------
