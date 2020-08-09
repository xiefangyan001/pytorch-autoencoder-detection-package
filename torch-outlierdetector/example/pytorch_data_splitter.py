"""the class for splitting the dataset in PyTorch
"""
# Author: Fangyan Xie <fangyan@mit.edu>

import torch


class TorchDataSplitter():
    def __init__(self, ratio=[0.7, 0.3]):
        '''
        ratio: the ratio of the amount of data in each subset, the
        sum of ratio should be equal to 1.
        '''
        self.ratio = ratio
        if sum(ratio) != 1:
            raise ValueError('the sum of ratio is not equal to 1')

    def split_random(self, X, y=None, random_seed=1):
        '''
        Split the data randomly.
        X: multi-dimensional point data
        y: optional, labels
        random_seed: the random seed for torch.Generator.manual_seed()
        return a list contains several torch tensor object, the order is
        as follows: [X_1, X_2,..., X_len(ratio), y_1, y_2,...,y_len(ratio)]
        '''
        if y is not None:
            if len(X) != len(y):
                raise RuntimeError('the sizes of X and y are different')
        x_size = len(X)
        split_list = [int(x_size * self.ratio[0])]
        p_sum = self.ratio[0]
        for i in range(len(self.ratio) - 1):
            p_sum += self.ratio[i + 1]
            split_list.append(int(x_size * p_sum) - sum(split_list))
        subset_list = torch.utils.data.random_split(X, split_list, generator=torch.Generator().manual_seed(random_seed))
        return_list = []
        for i in subset_list:
            return_list.append(X[i.indices])
        if y is not None:
            for i in subset_list:
                return_list.append(y[i.indices])
        return return_list

    def split_inorder(self, X, y=None):
        '''
        Split the data in order.
        X: multi-dimensional point data
        y: optional, labels
        return a list contains several torch tensor object, the order is
        as follows: [X_1, X_2,..., X_len(ratio), y_1, y_2,...,y_len(ratio)]
        '''
        if y is not None:
            if len(X) != len(y):
                raise RuntimeError('the sizes of X and y are different')
        x_size = len(X)
        split_list = [int(x_size * self.ratio[0])]
        p_sum = self.ratio[0]
        for i in range(len(self.ratio) - 1):
            p_sum += self.ratio[i + 1]
            split_list.append(int(x_size * p_sum) - sum(split_list))
        X_split = torch.split(X, split_list)
        if y is not None:
            y_split = torch.split(y, split_list)
        return_list = []
        for i in range(len(self.ratio)):
            return_list.append(X_split[i])
        if y is not None:
            for i in range(len(self.ratio)):
                return_list.append(y_split[i])
        return return_list
