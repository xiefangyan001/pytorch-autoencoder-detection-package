"""An easy implementing version applying AutoEncoder on Pytorch to
detect outliers.

For the version 0.0.0, we apply Adam as our optimization methods
and MSE to track the error during the training process.

What we can choose are {hidden layer, activation, batches, epochs}
and we only have 3 choices on the activation functions.
"""
# Author: Fangyan Xie <fangyan@mit.edu>


import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_curve, mean_squared_error


class AutoEncoder(nn.Module):
    def __init__(self, X, hidden_layer, activation):
        '''

        hidden_layer: a list contains the number of nodes
        in each hidden layer, notice do not have to include the
        input and output layer, since they can be inferred from
        the training data.

        activation: a list contains the activation functions
        between each layer, options include:
        {'relu','sigmoid','tanh'}

        In practice, we won't use this class directly, instead we
        will simply imply torch_AEOD
        '''
        super(AutoEncoder, self).__init__()
        encoder_list = []
        decoder_list = []
        last_size = len(X[0])
        for i in range((len(hidden_layer)+1) // 2):
            encoder_list.append(nn.Linear(last_size, hidden_layer[i]))
            decoder_list.append(nn.Linear(hidden_layer[i],last_size))
            last_size = hidden_layer[i]
        decoder_list = decoder_list[::-1].copy()

        encoder_layers = []
        decoder_layers = []
        for i in range(len(encoder_list)):
            encoder_layers.append(encoder_list[i])
            if activation[i] == 'relu':
                encoder_layers.append(nn.ReLU())
            if activation[i] == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            if activation[i] == 'tanh':
                encoder_layers.append(nn.Tanh())
        for i in range(len(decoder_list)):
            decoder_layers.append(decoder_list[i])
            if activation[i+len(activation) // 2] == 'relu':
                decoder_layers.append(nn.ReLU())
            if activation[i+len(activation) // 2] == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            if activation[i+len(activation) // 2] == 'tanh':
                decoder_layers.append(nn.Tanh())
        print(encoder_layers)
        print(decoder_layers)
        self.encoder = nn.Sequential(
            *encoder_layers
        )

        self.decoder = nn.Sequential(
            *decoder_layers
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class torch_AEOD():
    def __init__(self, hidden_layer, activation):
        '''

        hidden_layer: a list contains the number of nodes
        in each hidden layer, notice do not have to include the
        input and output layer, since they can be inferred from
        the training data.

        activation: a list contains the activation functions
        between each layer, options include:
        {'relu','sigmoid','tanh'}

        '''
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.model = None

    def train(self, X, epochs, batches, learning_rate):
        '''

        X: the training sample.
        epochs, batches, learning_rate are chosen by the researcher
        After training, we can use decision function to compute the anomaly
        scores for testing samples.
        '''
        self.model = AutoEncoder(X, self.hidden_layer, self.activation)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_func = nn.MSELoss()
        X = X.float()
        train_loader = Data.DataLoader(
            dataset=X,
            batch_size=batches,
            shuffle=True,
            num_workers=2,
        )
        for epoch in range(epochs):
            for step, x in enumerate(train_loader):
                b_x = x.view(-1, len(X[0]))
                b_y = x.view(-1, len(X[0]))

                encoded, decoded = self.model(b_x)

                loss = loss_func(decoded, b_y)  # mean square error
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()
                loss += loss.item()
            # compute the epoch training loss
            loss = loss / len(train_loader)
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


    def decision_function(self,X):
        '''

        X: the testing data that we need to compute scores, it can be different
        from the training samples.
        return a list, which contains the anomaly score for each test sample,
        the anomaly scores are computed by their reconstruction errors.
        '''
        X = X.float()
        score_list = []
        for i in range(len(X)):
            score_list.append(mean_squared_error(X[i].detach().numpy(),self.model(X)[1][i].detach().numpy()))
        return score_list

    def roc_curve(self, X, y):
        '''
        this could help us to do the HP tuning if we have their true labels.
        In financial world, the true label could be "whether it performs good or
        bad in the next period"
        X: the testing data
        y: their true labels
        return tpr, fpr and threshold for calculating those rate.
        '''

        X = X.float()
        y = list(y.detach().numpy())
        decision_score = self.decision_function(X)
        fpr, tpr, thresholds = roc_curve(y, decision_score, pos_label=1)
        return fpr, tpr, thresholds
