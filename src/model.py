import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.weight_norm as weight_norm

dropout_rate = .2

class metricHead(nn.Module):   # multi-class classifier
    def __init__(self, n_latent, n_hidden1, n_hidden2):
        super(metricHead, self).__init__()
        # self.n_input = n_input
        # n_hidden = 512

        self.fc1 = nn.Sequential(
                nn.Linear(n_latent, n_hidden1),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate)
            )
        self.fc2 = nn.Sequential(
                nn.Linear(n_hidden1, n_hidden2),
                # nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate)
            ) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=1)
        return x

class MSNet(nn.Module):   # multi-class classifier
    def __init__(self, n_input, n_latent, n_head1, n_head2, dropout_rate=0.2):
        super(MSNet, self).__init__()
        print('dropout_rate: ', dropout_rate)

        self.encoder = nn.Sequential(
                nn.Linear(n_input, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_head1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            )
        self.head = nn.Sequential(
                nn.Linear(n_head1, n_head1),
                nn.ReLU(inplace=True),
                nn.Linear(n_head1, n_head2)
                # nn.Dropout(p=dropout_rate)
            ) 

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x

class Net_encoder(nn.Module):
    def __init__(self, input_size, n_latent):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        # self.k = 64
        # self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, n_latent)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding

class Nonlinear_encoder(nn.Module):
    def __init__(self, input_size, n_latent, bn=False, dr=True):
        super(Nonlinear_encoder, self).__init__()
        self.input_size = input_size
        # self.k = 64
        # self.f = 64

        if bn:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate),
            )
        elif dr:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
            )


    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        # embedding = F.normalize(embedding)

        return embedding


class Net_cell(nn.Module):
    def __init__(self, n_latent, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(n_latent, num_of_class)
        )
        # self.cell = nn.weight_norm(nn.Linear(n_latent, num_of_class), name='weight')

    def forward(self, embedding):
        # embedding = F.normalize(embedding)
        cell_prediction = self.cell(embedding)

        return cell_prediction



class simple_classifier(nn.Module):   # multi-class classifier
    def __init__(self, n_latent, n_hidden, n_class):
        super(simple_classifier, self).__init__()
        # self.n_input = n_input
        # n_hidden = 512

        self.fc1 = nn.Sequential(
                nn.Linear(n_latent, n_class),
                # nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate)
            )
        # self.fc2 = nn.Sequential(
        #         nn.Linear(n_hidden, n_hidden),
        #         nn.ReLU(inplace=True),
        #         # nn.Dropout(p=dropout_rate)
        #     ) 
        # self.fc3 = nn.Sequential(
        #         nn.Linear(n_hidden, n_class),
        #         nn.ReLU(inplace=True),
        #     )

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find('Linear') != -1:
        layer.weight.data.normal_(0., 0.1)
        layer.bias.data.fill_(0.)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0.)
