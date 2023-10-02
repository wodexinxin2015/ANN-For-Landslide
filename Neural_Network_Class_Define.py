# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import scipy as sp
from torch.utils.data import Dataset
import torchmetrics as tm


# ----------------------------------------------------------------------------------------------------------------------
# Input --> Linear 1 --> Activation function 1 --> Linear 2 --> Activation function 2 --> Linear 3 --> Output
# ----------------------------------------------------------------------------------------------------------------------
class Neural_Network_Class(nn.Module):  # define the three-layer neural network
    # define the layer number of neural network
    def __init__(self, input_size, hidden_size, output_size, actifun_type1, actifun_type2):
        super(Neural_Network_Class, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        if actifun_type1 == 1:
            self.actfun1 = nn.Sigmoid()
        elif actifun_type1 == 2:
            self.actfun1 = nn.ReLU()
        elif actifun_type1 == 3:
            self.actfun1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        if actifun_type2 == 1:
            self.actfun2 = nn.Sigmoid()
        elif actifun_type2 == 2:
            self.actfun2 = nn.ReLU()
        elif actifun_type2 == 3:
            self.actfun2 = nn.Tanh()
        self.predict = nn.Linear(hidden_size, output_size)

    # define the forward propagation process
    def forward(self, input):
        out = self.linear1(input)
        out = self.actfun1(out)
        out = self.linear2(out)
        out = self.actfun2(out)
        out = self.predict(out)
        return out

    # define the training process
    def training_process(self, learn_r, bat_size, train_loop, optim_type, loss_type, data_dir, slice_num):
        # load data and label from file
        data_numpy = np.loadtxt(data_dir, dtype=np.float32, delimiter=',', skiprows=0)
        data_tensor = torch.from_numpy(data_numpy)
        features_o = data_tensor[:, :slice_num]
        labels_o = data_tensor[:, slice_num]
        # normalization
        feat_max, _ = torch.max(features_o, dim=0)
        feat_min, _ = torch.min(features_o, dim=0)
        label_max, _ = torch.max(labels_o, dim=0)
        label_min, _ = torch.min(labels_o, dim=0)
        features = (features_o - feat_min) / (feat_max - feat_min)
        labels = (labels_o - label_min) / (label_max - label_min)
        train_dataset = torch.utils.data.TensorDataset(features, labels)

        # define the optimizer
        if optim_type == 1:  # using the Stochastic Gradient Descent method
            optimizer = torch.optim.SGD(self.parameters(), learn_r)
        elif optim_type == 2:  # using the Adam method
            optimizer = torch.optim.Adam(self.parameters(), learn_r)
        elif optim_type == 3:  # using the RMSprop method
            optimizer = torch.optim.RMSprop(self.parameters(), learn_r)
        elif optim_type == 4:  # using the Adagrad method
            optimizer = torch.optim.Adagrad(self.parameters(), learn_r)
        print(optimizer)
        # define the loss function
        if loss_type == 1:
            criterion = nn.MSELoss()  # mean squared error function
        elif loss_type == 2:
            criterion = nn.CrossEntropyLoss()  # cross entropy error function
        elif loss_type == 3:
            criterion = nn.PoissonNLLLoss()  # PoissonNLLLoss function
        print(criterion)
        # define the array for the loss function
        loss_holder = []
        # set the loss as infinity: loss_value < pre_loss_value, save model
        loss_value = np.inf
        step = 0
        # loading the training data
        for t_id in range(train_loop):
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bat_size, shuffle=True)
            for batch_idx, (data, label) in enumerate(train_loader):
                # the result of forward process
                output = torch.squeeze(self.forward(data))
                # loss value
                loss = criterion(output, label)
                # back propagation: all the gradients are zero before back propagation.
                optimizer.zero_grad()
                loss.backward()
                # update parameters
                optimizer.step()
                # log the loss value
                if batch_idx % bat_size == 0:
                    step += 1
                    loss_holder.append([step, loss.detach().numpy() / bat_size])

                if batch_idx % bat_size == 0 and loss < loss_value:
                    torch.save(self.state_dict(), '0-model.pt')
                    loss_value = loss

        return loss_holder

    # define the training and testing process
    def training_testing_process(self, learn_r, bat_size, train_loop, optim_type, loss_type, train_num, test_num,
                                 train_data_dir, test_data_dir, slice_num):
        # load data and label from training file and testing file
        train_data_numpy = np.loadtxt(train_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
        train_data_tensor = torch.from_numpy(train_data_numpy)
        train_features_o = train_data_tensor[:, :slice_num]
        train_labels_o = train_data_tensor[:, slice_num]
        # normalization
        feat_max, _ = torch.max(train_features_o, dim=0)
        feat_min, _ = torch.min(train_features_o, dim=0)
        label_max, _ = torch.max(train_labels_o, dim=0)
        label_min, _ = torch.min(train_labels_o, dim=0)
        train_features = (train_features_o - feat_min) / (feat_max - feat_min)
        train_labels = (train_labels_o - label_min) / (label_max - label_min)
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)

        test_data_numpy = np.loadtxt(test_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
        test_data_tensor = torch.from_numpy(test_data_numpy)
        test_features = test_data_tensor[:, :slice_num]
        test_labels = test_data_tensor[:, slice_num]
        test_features = (test_features - feat_min) / (feat_max - feat_min)
        test_labels = (test_labels - label_min) / (label_max - label_min)
        test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

        # define the optimizer
        if optim_type == 1:  # using the Stochastic Gradient Descent method
            optimizer = torch.optim.SGD(self.parameters(), learn_r)
        elif optim_type == 2:  # using the Adam method
            optimizer = torch.optim.Adam(self.parameters(), learn_r)
        elif optim_type == 3:  # using the RMSprop method
            optimizer = torch.optim.RMSprop(self.parameters(), learn_r)
        elif optim_type == 4:  # using the Adagrad method
            optimizer = torch.optim.Adagrad(self.parameters(), learn_r)
        print(optimizer)
        # define the loss function
        if loss_type == 1:
            criterion = nn.MSELoss()  # mean squared error function
        elif loss_type == 2:
            criterion = nn.CrossEntropyLoss()  # cross entropy error function
        elif loss_type == 3:
            criterion = nn.PoissonNLLLoss()   # PoissonNLLLoss function
        print(criterion)
        # error function
        Accuracy_Fun_1 = tm.PearsonCorrCoef()
        Accuracy_Fun_2 = tm.CosineSimilarity()
        # define the array for the loss function and similarity score
        loss_holder = []
        simu_score_train = []
        simu_score_test = []
        # set the loss as infinity: loss_value < pre_loss_value, save model
        loss_value = np.inf
        step = 0
        # loading the training data
        for t_id in range(train_loop):
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bat_size, shuffle=True)
            for batch_idx_1, (data_1, label_1) in enumerate(train_loader):
                # the result of forward process
                output_1 = torch.squeeze(self.forward(data_1))
                # loss value
                loss = criterion(output_1, label_1)
                # back propagation: all the gradients are zero before back propagation.
                optimizer.zero_grad()
                loss.backward()
                # update parameters
                optimizer.step()
                # log the loss value
                if batch_idx_1 % bat_size == 0:
                    step += 1
                    loss_holder.append([step, loss.detach().numpy() / bat_size])

                if batch_idx_1 % bat_size == 0 and loss < loss_value:
                    torch.save(self.state_dict(), '0-model.pt')
                    loss_value = loss

            # calculating the similarity score for the training data
            train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=train_num, shuffle=False)
            for batch_idx_2, (data_2, label_2) in enumerate(train_loader_all):
                # the result of forward process
                output_2 = torch.squeeze(self.forward(data_2))
                score_train_per = Accuracy_Fun_1(label_2, output_2)
                score_train_cos = Accuracy_Fun_2(label_2, output_2)
                simu_score_train.append([t_id, score_train_per.detach().numpy(), score_train_cos.detach().numpy()])

            # calculating the similarity score for the testing data
            test_loader_all = torch.utils.data.DataLoader(test_dataset, batch_size=test_num, shuffle=False)
            for batch_idx_3, (data_3, label_3) in enumerate(test_loader_all):
                # the result of forward process
                output_3 = torch.squeeze(self.forward(data_3))
                score_test_per = Accuracy_Fun_1(label_3, output_3)
                score_test_cos = Accuracy_Fun_2(label_3, output_3)
                simu_score_test.append([t_id, score_test_per.detach().numpy(), score_test_cos.detach().numpy()])

        return loss_holder, simu_score_train, simu_score_test, feat_max, feat_min, label_max, label_min

    # define the process of loading model state and predict new data
    def model_load_predict(self, pred_data_dir, pred_result_dir):
        # loading the model parameters
        self.load_state_dict(torch.load('0-model.pt'))
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
            pred_result = self.forward(torch.unsqueeze(pred_data_tensor, dim=1))
            pred_result = pred_result * (label_max - label_min) + label_min
            pred_tensor = torch.cat([torch.unsqueeze(pred_data_tensor_o, dim=1), pred_result], 1)
        else:
            pred_result = self.forward(pred_data_tensor)
            pred_result = pred_result * (label_max - label_min) + label_min
            pred_tensor = torch.cat([pred_data_tensor_o, pred_result], 1)
        np.savetxt(pred_result_dir, pred_tensor.detach().numpy(), fmt='%.6f', delimiter=',')

    # define the process of loading model state and predict new data
    def random_para_predict(self, pred_result_dir, mean_var, std_var, coeff_var, mcs_times, dis_type):
        # loading the model parameters
        self.load_state_dict(torch.load('0-model.pt'))
        feat_max = torch.load('2-feat-max.pth')
        feat_min = torch.load('2-feat-min.pth')
        label_max = torch.load('2-label-max.pth')
        label_min = torch.load('2-label-min.pth')
        # generate random parameters
        if dis_type == 2:  # log normal distribution
            temp_mean = np.array(mean_var)
            temp_std = np.array(std_var)
            mean_var_log = np.log(temp_mean * temp_mean / np.sqrt(temp_std + temp_mean * temp_mean))
            std_var_log = np.sqrt(np.log(temp_std / temp_mean * temp_mean + 1))
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
            pred_result = self.forward(torch.unsqueeze(pred_data_tensor.to(torch.float32), dim=1))
            pred_result = pred_result * (label_max - label_min) + label_min
            pred_tensor = torch.cat([torch.unsqueeze(pred_data_tensor_o, dim=1), pred_result], 1)
        else:
            pred_result = self.forward(pred_data_tensor.to(torch.float32))
            pred_result = pred_result * (label_max - label_min) + label_min
            pred_tensor = torch.cat([pred_data_tensor_o, pred_result], 1)
        np.savetxt(pred_result_dir, pred_tensor.detach().numpy(), fmt='%.6f', delimiter=',')
