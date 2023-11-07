# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from torch.utils.data import Dataset
import torchmetrics as tm


# ----------------------------------------------------------------------------------------------------------------------
# define the training and testing process
def training_testing_process(net, learn_r, weight_d, bat_size, train_loop, optim_type, loss_type,
                             train_data_dir, test_data_dir, slice_num, output_size_t):
    # load data and label from training file and testing file
    train_data_numpy = np.loadtxt(train_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
    train_data_tensor = torch.from_numpy(train_data_numpy)
    train_features_o = train_data_tensor[:, :slice_num]
    train_labels_o = train_data_tensor[:, slice_num:]
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
    test_labels = test_data_tensor[:, slice_num:]
    test_features = (test_features - feat_min) / (feat_max - feat_min)
    test_labels = (test_labels - label_min) / (label_max - label_min)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    # define the optimizer
    if optim_type == 1:  # using the Stochastic Gradient Descent method
        optimizer = torch.optim.SGD(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 2:  # using the Adam method
        optimizer = torch.optim.Adam(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 3:  # using the RMSprop method
        optimizer = torch.optim.RMSprop(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 4:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(net.parameters(), learn_r, weight_decay=weight_d)
    print(optimizer)
    # define the loss function
    if loss_type == 1:
        criterion = nn.MSELoss(reduction='mean')  # mean squared error function
    elif loss_type == 2:
        criterion = nn.CrossEntropyLoss()  # cross entropy error function
    elif loss_type == 3:
        criterion = nn.PoissonNLLLoss()  # PoissonNLLLoss function
    print(criterion)
    # error function
    Accuracy_Fun_1 = tm.PearsonCorrCoef(num_outputs=output_size_t)
    Accuracy_Fun_2 = tm.R2Score(num_outputs=output_size_t, multioutput='raw_values')
    # define the array for the loss function and similarity score
    loss_holder_train = []
    loss_holder_test = []
    simu_score_train = []
    simu_score_test = []
    # set the loss as infinity: loss_value < pre_loss_value, save model
    loss_value = np.inf
    # loading the training data
    for t_id in range(train_loop):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bat_size, shuffle=True)
        for batch_idx_1, (data_1, label_1) in enumerate(train_loader):
            # the result of forward process
            output_1 = net.forward(data_1)
            # loss value
            loss = criterion(output_1, label_1)
            # back propagation: all the gradients are zero before back propagation.
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()

        # calculating the similarity score for the training data
        train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        for batch_idx_2, (data_2, label_2) in enumerate(train_loader_all):
            # the result of forward process
            output_2 = net.forward(data_2)
            if output_size_t == 1:
                score_train_per = Accuracy_Fun_1(torch.squeeze(output_2), torch.squeeze(label_2))
                score_train_cos = Accuracy_Fun_2(torch.squeeze(output_2), torch.squeeze(label_2))
            else:
                score_train_per = Accuracy_Fun_1(output_2, label_2)
                score_train_cos = Accuracy_Fun_2(output_2, label_2)
            loss_1 = criterion(output_2, label_2)
            loss_holder_train.append([t_id, criterion(output_2, label_2).detach().numpy()])
            if loss_1 < loss_value:
                torch.save(net.state_dict(), '0-model.pt')
                loss_value = loss_1
            simu_score_train.append([t_id, score_train_per.detach().numpy(), score_train_cos.detach().numpy()])

        # calculating the similarity score for the testing data
        test_loader_all = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        for batch_idx_3, (data_3, label_3) in enumerate(test_loader_all):
            # the result of forward process
            output_3 = net.forward(data_3)
            loss_holder_test.append([t_id, criterion(output_3, label_3).detach().numpy()])
            if output_size_t == 1:
                score_test_per = Accuracy_Fun_1(torch.squeeze(output_3), torch.squeeze(label_3))
                score_test_cos = Accuracy_Fun_2(torch.squeeze(output_3), torch.squeeze(label_3))
            else:
                score_test_per = Accuracy_Fun_1(output_3, label_3)
                score_test_cos = Accuracy_Fun_2(output_3, label_3)
            simu_score_test.append([t_id, score_test_per.detach().numpy(), score_test_cos.detach().numpy()])

    return loss_holder_train, loss_holder_test, simu_score_train, simu_score_test, feat_max, feat_min, \
           label_max, label_min


# ----------------------------------------------------------------------------------------------------------------------
# define the incremental train process
def training_testing_incremental(net, learn_r, weight_d, bat_size, train_loop, optim_type, loss_type,
                                 train_data_dir, test_data_dir, slice_num, feat_max, feat_min, label_max,
                                 label_min, output_size_t):
    # load data and label from training file and testing file
    train_data_numpy = np.loadtxt(train_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
    train_data_tensor = torch.from_numpy(train_data_numpy)
    train_features_o = train_data_tensor[:, :slice_num]
    train_labels_o = train_data_tensor[:, slice_num:]
    # normalization
    train_features = (train_features_o - feat_min) / (feat_max - feat_min)
    train_labels = (train_labels_o - label_min) / (label_max - label_min)
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)

    test_data_numpy = np.loadtxt(test_data_dir, dtype=np.float32, delimiter=',', skiprows=0)
    test_data_tensor = torch.from_numpy(test_data_numpy)
    test_features = test_data_tensor[:, :slice_num]
    test_labels = test_data_tensor[:, slice_num:]
    test_features = (test_features - feat_min) / (feat_max - feat_min)
    test_labels = (test_labels - label_min) / (label_max - label_min)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    # define the optimizer
    if optim_type == 1:  # using the Stochastic Gradient Descent method
        optimizer = torch.optim.SGD(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 2:  # using the Adam method
        optimizer = torch.optim.Adam(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 3:  # using the RMSprop method
        optimizer = torch.optim.RMSprop(net.parameters(), learn_r, weight_decay=weight_d)
    elif optim_type == 4:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(net.parameters(), learn_r, weight_decay=weight_d)
    print(optimizer)
    # define the loss function
    if loss_type == 1:
        criterion = nn.MSELoss(reduction='mean')  # mean squared error function
    elif loss_type == 2:
        criterion = nn.CrossEntropyLoss()  # cross entropy error function
    elif loss_type == 3:
        criterion = nn.PoissonNLLLoss()  # PoissonNLLLoss function
    print(criterion)
    # error function
    Accuracy_Fun_1 = tm.PearsonCorrCoef(num_outputs=output_size_t)
    Accuracy_Fun_2 = tm.R2Score(num_outputs=output_size_t, multioutput='raw_values')
    # define the array for the loss function and similarity score
    loss_holder_train = []
    loss_holder_test = []
    simu_score_train = []
    simu_score_test = []
    # set the loss as infinity: loss_value < pre_loss_value, save model
    loss_value = np.inf
    # loading the training data
    for t_id in range(train_loop):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bat_size, shuffle=True)
        for batch_idx_1, (data_1, label_1) in enumerate(train_loader):
            # the result of forward process
            output_1 = net.forward(data_1)
            # loss value
            loss = criterion(output_1, label_1)
            # back propagation: all the gradients are zero before back propagation.
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()

        # calculating the similarity score for the training data
        train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        for batch_idx_2, (data_2, label_2) in enumerate(train_loader_all):
            # the result of forward process
            output_2 = net.forward(data_2)
            if output_size_t == 1:
                score_train_per = Accuracy_Fun_1(torch.squeeze(output_2), torch.squeeze(label_2))
                score_train_cos = Accuracy_Fun_2(torch.squeeze(output_2), torch.squeeze(label_2))
            else:
                score_train_per = Accuracy_Fun_1(output_2, label_2)
                score_train_cos = Accuracy_Fun_2(output_2, label_2)
            loss_1 = criterion(output_2, label_2)
            loss_holder_train.append([t_id, criterion(output_2, label_2).detach().numpy()])
            if loss_1 < loss_value:
                torch.save(net.state_dict(), '0-model.pt')
                loss_value = loss_1
            simu_score_train.append([t_id, score_train_per.detach().numpy(), score_train_cos.detach().numpy()])

        # calculating the similarity score for the testing data
        test_loader_all = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        for batch_idx_3, (data_3, label_3) in enumerate(test_loader_all):
            # the result of forward process
            output_3 = net.forward(data_3)
            loss_holder_test.append([t_id, criterion(output_3, label_3).detach().numpy()])
            if output_size_t == 1:
                score_test_per = Accuracy_Fun_1(torch.squeeze(output_3), torch.squeeze(label_3))
                score_test_cos = Accuracy_Fun_2(torch.squeeze(output_3), torch.squeeze(label_3))
            else:
                score_test_per = Accuracy_Fun_1(output_3, label_3)
                score_test_cos = Accuracy_Fun_2(output_3, label_3)
            simu_score_test.append([t_id, score_test_per.detach().numpy(), score_test_cos.detach().numpy()])

    return loss_holder_train, loss_holder_test, simu_score_train, simu_score_test




