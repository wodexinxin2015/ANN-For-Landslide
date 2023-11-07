# ----------------------------------------------------------------------------------------------------------------------
# Neural_Network_Class for the training of geotechnical data
# By Weijie Zhang, Hohai University
# ----------------------------------------------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim
import torchmetrics as tm
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss


# ----------------------------------------------------------------------------------------------------------------------
class Neural_Network_Class(nn.Module):  # define the three-layer neural network
    # define the layer number of neural network
    def __init__(self, input_size_n, hidden_size_n, output_size_n):
        super(Neural_Network_Class, self).__init__()
        # layer 1
        self.linear1 = nn.Linear(input_size_n, hidden_size_n)
        # self.actfun1 = nn.Sigmoid()
        self.actfun1 = nn.ReLU()
        # self.actfun1 = nn.Tanh()
        # layer 2
        self.linear2 = nn.Linear(hidden_size_n, hidden_size_n)
        # self.actfun2 = nn.Sigmoid()
        self.actfun2 = nn.ReLU()
        # self.actfun2 = nn.Tanh()
        # layer 3
        self.predict = nn.Linear(hidden_size_n, output_size_n)

    # define the forward propagation process
    def forward(self, input_n):
        out = self.linear1(input_n)
        out = self.actfun1(out)
        out = self.linear2(out)
        out = self.actfun2(out)
        out = self.predict(out)
        return out


# ----------------------------------------------------------------------------------------------------------------------
def get_k_fold_data(k, i, train_feat, train_label):
    assert k > 1
    fold_size = len(train_feat) / k

    feat_train = None
    feat_valid = None
    label_train = None
    label_valid = None
    for j in range(k):
        idx = slice(int(j * fold_size), int((j + 1) * fold_size))
        # idx are the validation group
        feat_part = torch.FloatTensor(train_feat[idx, :])
        label_part = torch.FloatTensor(train_label[idx])
        if j == i:
            feat_valid = feat_part
            label_valid = label_part
        elif feat_train is None:
            feat_train = feat_part
            label_train = label_part
        else:
            feat_train = torch.cat([feat_train, feat_part], dim=0)
            label_train = torch.cat([label_train, label_part], dim=0)
    return feat_train, feat_valid, label_train, label_valid


# ----------------------------------------------------------------------------------------------------------------------
def k_fold(input_size_k, hidden_size_k, output_size_k, k_k, train_feat, train_label, num_epoch_k, lr_rate_k, weight_d_k,
           bat_size_k):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    e_s_temp = 0

    for i in range(k_k):
        feat_train, feat_valid, label_train, label_valid = get_k_fold_data(k_k, i, train_feat, train_label)
        data_train = torch.utils.data.TensorDataset(feat_train, label_train)
        data_valid = torch.utils.data.TensorDataset(feat_valid, label_valid)
        net = Neural_Network_Class(input_size_k, hidden_size_k, output_size_k)
        # train the network
        train_loss, valid_loss, train_acc, valid_acc, e_stop = train_process \
            (net, data_train, data_valid, num_epoch_k, lr_rate_k, weight_d_k, bat_size_k, output_size_k)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += valid_acc[-1]
        e_s_temp += e_stop

    print('\n', '#' * 10, 'Result of k-fold cross validation', '#' * 10)
    print('average train loss:{:.4f}, average train accuracy:{}'.format(train_loss_sum / k_k, train_acc_sum / k_k))
    print('average valid loss:{:.4f}, average valid accuracy:{}'.format(valid_loss_sum / k_k, valid_acc_sum / k_k))
    return train_loss_sum / k_k, valid_loss_sum / k_k, train_acc_sum / k_k, valid_acc_sum / k_k, e_s_temp / k_k


# ----------------------------------------------------------------------------------------------------------------------
def train_process(net, data_train, data_valid, num_epochs, lr_rate_t, wd_t, bat_size_t, output_size_t):
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []

    optim_type = 3
    loss_type = 1
    # define the optimizer
    if optim_type == 1:  # using the Stochastic Gradient Descent method
        optimizer = torch.optim.SGD(net.parameters(), lr_rate_t, weight_decay=wd_t)
    elif optim_type == 2:  # using the Adam method
        optimizer = torch.optim.Adam(net.parameters(), lr_rate_t, weight_decay=wd_t)
    elif optim_type == 3:  # using the RMSprop method
        optimizer = torch.optim.RMSprop(net.parameters(), lr_rate_t, weight_decay=wd_t)
    elif optim_type == 4:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(net.parameters(), lr_rate_t, weight_decay=wd_t)
    # define the loss function
    if loss_type == 1:
        criterion = nn.MSELoss(reduction='mean')  # mean squared error function
    elif loss_type == 2:
        criterion = nn.CrossEntropyLoss()  # cross entropy error function
    elif loss_type == 3:
        criterion = nn.PoissonNLLLoss()  # PoissonNLLLoss function
    # define the early stopping function
    early_stopper = EarlyStopping(10)
    acc_fun = tm.R2Score(num_outputs=output_size_t, multioutput='raw_values')

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=bat_size_t, shuffle=True)
    for epoch in range(num_epochs):
        # train process with mini-batch training
        for batch_idx, (data, label) in enumerate(train_loader):
            # the result of forward process
            output = net.forward(data)
            # loss value
            loss = criterion(output, label)
            # back propagation: all the gradients are zero before back propagation.
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()

        # the accuracy of train set
        train_loader_1 = torch.utils.data.DataLoader(data_train, batch_size=len(data_train), shuffle=False)
        score_per = torch.zeros(output_size_t).reshape([1, output_size_t])
        for batch_idx_1, (data_1, label_1) in enumerate(train_loader_1):
            # the result of forward process
            net.eval()
            output_1 = net.forward(data_1)
            train_loss.append(criterion(output_1, label_1).detach().numpy())
            if output_size_t == 1:
                score_per += acc_fun(torch.squeeze(label_1), torch.squeeze(output_1))
            else:
                score_per += acc_fun(label_1, output_1)
        train_acc.append(score_per.detach().numpy() / len(train_loader_1))

        # the accuracy of valid set
        valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=len(data_valid), shuffle=False)
        score_per = torch.zeros(output_size_t).reshape([1, output_size_t])
        for batch_idx_2, (data_2, label_2) in enumerate(valid_loader):
            # the result of forward process
            net.eval()
            output_2 = net.forward(data_2)
            valid_loss.append(criterion(output_2, label_2).detach().numpy())
            if output_size_t == 1:
                score_per += acc_fun(torch.squeeze(label_2), torch.squeeze(output_2))
            else:
                score_per += acc_fun(label_2, output_2)
        valid_acc.append(score_per.detach().numpy() / len(valid_loader))
        stop = epoch

        # early stopping
        early_stopper(valid_loss[-1], net)
        if early_stopper.early_stop:
            break
    return train_loss, valid_loss, train_acc, valid_acc, stop


# ----------------------------------------------------------------------------------------------------------------------
input_size = 2
output_size = 1
slice_num = 2
# ----------------------------------------------------------------------------------------------------------------------
k = 10
epoch_num = 300
# ----------------------------------------------------------------------------------------------------------------------
# hyperparameters
hidden_size_range = np.arange(3, 13, 1).tolist()
lr_rate_range = np.arange(0.001, 0.014, 0.001).tolist()
weight_decay_range = np.arange(0.001, 0.003, 0.001).tolist()
bat_size_range = np.arange(4, 8, 1).tolist()
# ----------------------------------------------------------------------------------------------------------------------
if os.sep == "/":  # linux platform
    train_data_dir = r'./train-data/train-data.csv'
else:  # windows platform
    train_data_dir = r'.\\train-data\\train-data.csv'
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
# ----------------------------------------------------------------------------------------------------------------------
h_size, lr, w_d, b_s, t_loss, v_loss, t_acc, v_acc, e_s = 0, 0, 0, 0, 0, 0, 0, 0, 0
mean_acc_pre = -10000.0
for hidden_size in hidden_size_range:
    for lr_rate in lr_rate_range:
        for weight_decay in weight_decay_range:
            for bat_size in bat_size_range:
                print("--------------------------------------------------------------------------------")
                print(
                    'hidden_size:{:2d}, learn_rate:{:.4f}, weight_decay:{:.4f}, bat_size:{:2d}'.format(hidden_size,
                                                                                                       lr_rate,
                                                                                                       weight_decay,
                                                                                                       bat_size))
                print("----------------------------------------")
                t_loss_1, v_loss_1, t_acc_1, v_acc_1, e_s_1 = k_fold(input_size, hidden_size, output_size, k,
                                                                     train_features, train_labels, epoch_num, lr_rate,
                                                                     weight_decay, bat_size)
                t_loss_2, v_loss_2, t_acc_2, v_acc_2, e_s_2 = k_fold(input_size, hidden_size, output_size, k,
                                                                     train_features, train_labels, epoch_num, lr_rate,
                                                                     weight_decay, bat_size)
                t_loss_3, v_loss_3, t_acc_3, v_acc_3, e_s_3 = k_fold(input_size, hidden_size, output_size, k,
                                                                     train_features, train_labels, epoch_num, lr_rate,
                                                                     weight_decay, bat_size)
                t_loss_4, v_loss_4, t_acc_4, v_acc_4, e_s_4 = k_fold(input_size, hidden_size, output_size, k,
                                                                     train_features, train_labels, epoch_num, lr_rate,
                                                                     weight_decay, bat_size)                                                                    
                mean_acc = (v_acc_1 + v_acc_2 + v_acc_3 + v_acc_4 ) * 0.25
                if np.mean(mean_acc) > np.mean(mean_acc_pre):
                    h_size = hidden_size
                    lr = lr_rate
                    w_d = weight_decay
                    b_s = bat_size
                    t_loss = (t_loss_2 + t_loss_2 + t_loss_3 + t_loss_4) * 0.25
                    v_loss = (v_loss_2 + v_loss_2 + v_loss_3 + v_loss_4) * 0.25
                    t_acc = (t_acc_1 + t_acc_2 + t_acc_3 + t_acc_4) * 0.25
                    v_acc = mean_acc
                    e_s = e_s_2
                    mean_acc_pre = mean_acc

# print the result
print("----------------------------------------")
print("The best set of hyperparameters:")
print('hidden_size:{:2d}, learn_rate:{:.4f}, weight_decay:{:.4f}, bat_size:{:2d}'.format(h_size, lr, w_d, b_s))
print('early stopping epoch:{:.1f}'.format(e_s))
print('average train loss:{:.4f}, average train accuracy:{}'.format(t_loss, t_acc))
print('average valid loss:{:.4f}, average valid accuracy:{}'.format(v_loss, v_acc))
# ----------------------------------------------------------------------------------------------------------------------
