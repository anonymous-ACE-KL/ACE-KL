import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import math
from tqdm import tqdm
from sklearn import metrics
import pdb

def softplus(x): # Smooth Relu
    return np.log(1 + np.exp(x))

def weighted_mse_loss(input,target,weights):
    out = (torch.squeeze(input)-target)**2
    loss = torch.mean(out * weights) # or sum over whatever dimensions

    return loss

def mse_loss(input,target):
    out = (torch.squeeze(input)-target)**2
    # pdb.set_trace()
    loss = torch.mean(out) # or sum over whatever dimensions
    return loss


'''
CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, lmd = 1.0/5000):
        super(LinearRegressionModel, self).__init__()
        self.lmd = lmd
        self.weight = torch.nn.Parameter(torch.zeros(1, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_X):
        """
        linear part
        """
        linear_part = F.linear(input_X, self.weight, self.bias)

        return linear_part

    def cal_loss(self, input_X, input_Y, sample_weights):
        linear_part = self.forward(input_X)
        mse = weighted_mse_loss(linear_part, input_Y, sample_weights)
        regularzizer = self.lmd * torch.sum(self.weight**2)

        loss = mse + regularzizer

        return loss


def main():

    """
    load data
    """
    X = np.load(sys.argv[1])
    X = np.where(X == 0, -1, 1).astype(np.float32)
    # print X
    y = np.load(sys.argv[2])
    y = -np.log(y).astype(np.float32)

    x_train = X

    y_train = y

    input_dim = X.shape[1]

    original_datum = x_train[0]
    distances = metrics.pairwise_distances(x_train,
                                            original_datum.reshape(1, -1),
                                            metric='euclidean'
                                            ).ravel()

    kernel_width = np.sqrt(original_datum.shape[0]) * .75
    kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    sample_weights_train = kernel_fn(distances)


    '''
    INSTANTIATE MODEL CLASS
    '''

    model = LinearRegressionModel(input_dim)

    rng = np.random.RandomState(12345)

    model.cuda()


    learning_rate = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 50000
    for epoch in tqdm(range(epochs)):
        epoch += 1

        idxs = rng.permutation(len(y_train))
        # pdb.set_trace()
        x_train, y_train, sample_weights_train = x_train[idxs], y_train[idxs], sample_weights_train[idxs]

        if torch.cuda.is_available():
            vx = Variable(torch.from_numpy(x_train).cuda())

        if torch.cuda.is_available():
            vy = Variable(torch.from_numpy(y_train).cuda())

        if torch.cuda.is_available():
            vsample_weights = Variable(torch.from_numpy(sample_weights_train).cuda())

        optimizer.zero_grad()

        loss = model.cal_loss(vx, vy, vsample_weights)
        # print('loss {}'.format(loss.data[0]))
        # loss = criterion(outputs, labels)

        loss.backward()


        optimizer.step()


        # print('epoch {}, loss {}'.format(epoch, loss.data[0]))
    print('loss {}'.format(loss.data[0]))
    print model.weight.cpu().data.numpy()
    w = np.squeeze(model.weight.cpu().data.numpy())
    original_row_data = X[0]
    components = w*original_row_data
    print components
    contributions = softplus(components)/np.sum(softplus(components))
    print sorted(zip(range(input_dim), contributions),
                            key=lambda x: x[1],
                            reverse=True)


if __name__ == '__main__':
    main()
