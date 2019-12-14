import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score

train_size = 2000
test_size = 8000

batch_size = 512
epochs = 250
verbose = True

threshold = 0.0005

sample_size = 1000

class Model:
    def __init__(self, net, x, y):
        self.x = x.values
        self.y = y.values
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, train_size = train_size, test_size = test_size, random_state=0)
        self.net = net
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, criterion):
        x_tensor = torch.tensor(self.train_x).float()
        y_tensor = torch.tensor(self.train_y).float()

        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # make model
        self.net.to(self.device)

        optimizer = optim.Adam(self.net.parameters())

        # train model
        running_loss = 0.0
        done_flag = False
        for epoch in range(epochs):  # loop over the dataset multiple times
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(x_batch)
                loss = criterion(outputs, y_batch.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch % 5 == 0):
                if verbose:
                    print('[%d] loss: %.3f' % (epoch, running_loss / 5))
                if running_loss < threshold:
                    if done_flag:
                        print("Terminated Early. 10 successive epochs all have 0 loss")
                        break
                    done_flag = True
                else:
                    done_flag = False
                    running_loss = 0.0

        print('Finished Training')

        return self.net

    def _auc(self, set_1, set_2):
        train_sample = set_1[np.random.choice(set_1.shape[0], size = sample_size)]
        test_sample = set_2[np.random.choice(set_2.shape[0], size = sample_size)]
        train_confidence = torch.max(torch.exp(self.net(torch.Tensor(train_sample).to(self.device)).detach()), axis=1)[0]
        test_confidence = torch.max(torch.exp(self.net(torch.Tensor(test_sample).to(self.device)).detach()), axis=1)[0]

        actual = [1 for i in range(sample_size)] + [0 for i in range(sample_size)]
        estimated = torch.cat([train_confidence, test_confidence])
        return roc_auc_score(torch.Tensor(actual), estimated.cpu())
    
    def auc(self):
        return self._auc(self.train_x, self.test_x)
    
    def auc_by_distance(self, distance_func, group_size = 1):
        distances = self.distances(distance_func)
        groups = {}
        for (features, dist) in zip(self.test_x, distances):
            dist = dist//group_size * group_size
            if dist not in groups.keys():
                groups[dist] = []
            groups[dist].append(features)
        keys = sorted([key for key in groups.keys()])
        aucs = [self._auc(self.train_x, np.array(groups[key])) for key in keys]
        
        return (keys, aucs)
    
    dists = {}
    def distances(self, distance_func):
        key = hash((hash(self.test_x.tostring()), hash(self.train_x.tostring()), hash(distance_func)))
        if key not in Model.dists:
            Model.dists[key] = distance_func(self.test_x, self.train_x)
        return Model.dists[key]
    