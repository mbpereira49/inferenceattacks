import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn

import nets

# from utils import progress_bar

class Model:
    def __init__(self, random_state=0):
        torch.manual_seed(0)
        np.random.seed(0)

        self.random_state = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        raise NotImplemented

    def build_net(self):
        raise NotImplemented

    def load_net(self, net):
        """
        Lets you load a custom net not specified in nets.py
        """
        self.net = net
        
    def train(self, criterion, batch_size=512, epochs=400, verbose=True, early_stop_loss_threshold = 0.0005):
        """
        Trains self.net based on criterion loss function
        """

        self.train_loader = DataLoader(dataset=self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.train_x = np.stack([data[0] for data in self.trainset])
        self.test_x = np.stack([data[0] for data in self.testset])

        # make model
        self.net.to(self.device)
        self.net.train()  # training mode

        optimizer = optim.Adam(self.net.parameters())

        # train model
        running_loss = 0.0
        done_flag = False

        train_errors = []
        test_errors = []
        this_aucs = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            correct = 0
            total = 0
            for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
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
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

            with torch.no_grad():
                train_error = 1 - correct / total
                test_error = self.test_error()
                this_auc = self.auc(big=True)
                print("{0:.3f}, {1:.3f}, {2:.3f}".format(train_error, test_error, this_auc))

                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            if (epoch % 5 == 0):
                if verbose:
                    print('[%d] loss: %.3f' % (epoch, running_loss))
                if running_loss < early_stop_loss_threshold:
                    if done_flag:
                        print("Terminated Early. 10 successive epochs all have 0 loss")
                        break
                    done_flag = True
                else:
                    done_flag = False
                    running_loss = 0.0

        print('Finished Training')

        return train_errors, test_errors, this_aucs
        
    # from PyTorch tutorial
    def error(self, dataset):
        correct, total = 0, 0
        for data in dataset:
            images, labels = data
            outputs = self.net(images.to(self.device))
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum().item()
        return 1 - correct/total

    def train_error(self):
        return self.error(self.train_loader)
    def test_error(self):
        return self.error(self.test_loader)
    


    def _auc(self, set_1, set_2, big = False):
        with torch.no_grad():
            if big:
                train_confidence = []
                for elt in set_1:
                    train_confidence.append(torch.max(torch.exp(self.net(torch.Tensor([elt]).to(self.device)).detach()), axis=1)[0])
                train_confidence = torch.tensor(train_confidence)
                test_confidence = []
                for elt in set_2:
                    test_confidence.append(torch.max(torch.exp(self.net(torch.Tensor([elt]).to(self.device)).detach()), axis=1)[0])
                test_confidence = torch.tensor(test_confidence)
            else:
                train_confidence = torch.max(torch.exp(self.net(torch.Tensor(set_1).to(self.device)).detach()), axis=1)[0]
                test_confidence = torch.max(torch.exp(self.net(torch.Tensor(set_2).to(self.device)).detach()), axis=1)[0]

            actual = [1 for i in range(set_1.shape[0])] + [0 for i in range(set_2.shape[0])]
            estimated = torch.cat([train_confidence, test_confidence])
            return roc_auc_score(torch.Tensor(actual), estimated.cpu())
    
    def auc(self, big = False):
        return self._auc(self.train_x, self.test_x, big)
    
    def auc_by_distance(self, distance_func, group_size = 1):
        """
        returns tuple of distances and aucs of test data, useful for generating curve in paper
          - def distance_func(point, set). point is usually test set, set is training data
        """
        distances = self.distances(distance_func)
        distances = distance_func(self.test_x, self.train_x)
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
        key = hash((hash(self.trainset), hash(self.train_x.tostring()), hash(distance_func)))
        if key not in self.dists:
            Model.dists[key] = distance_func(self.test_x, self.train_x)
        return self.dists[key]


class PurchaseModel(Model):
    def __init__(self, n_labels, random_state=0):
        super().__init__(random_state)
        self.n_labels = n_labels

    def load_data(self, train_size=0.2, test_size=0.8, p_noise=0.0):
        print(f'Loading {self.n_labels}-Purchase from GitHub: {train_size} train, {test_size} test.')
        df_url = 'https://raw.githubusercontent.com/mbpereira49/inferenceattacks/master/data/df.csv'
        lab_url = 'https://raw.githubusercontent.com/mbpereira49/inferenceattacks/master/data/labels.csv'
        
        # read data
        x = pd.read_csv(df_url).drop('id', axis=1).values
        y = pd.read_csv(lab_url).drop('id', axis=1).astype(int)
        y.columns = [2, 10, 20, 50, 100]
        y = y[self.n_labels].values

        self.input_dim = x.shape[1]

        if (p_noise > 0):
            ind = np.random.choice(len(y), int(p_noise*len(y)), replace=False)
            for i in ind:
                curr = y[i]
                other_labels = np.concatenate([np.array(range(0, curr)), np.array(range(curr, self.n_labels))])
                y[i] = np.random.choice(other_labels)
        
        # split data
        train_x, test_x, train_y, test_y = train_test_split(x, y,
            train_size = train_size, 
            test_size = test_size, 
            random_state=self.random_state)
        x_tensor = torch.tensor(train_x).float()
        y_tensor = torch.tensor(train_y).float()
        x_test_tensor = torch.tensor(test_x).float()
        y_test_tensor = torch.tensor(test_y).float()

        self.trainset = TensorDataset(x_tensor, y_tensor)
        self.testset = TensorDataset(x_test_tensor, y_test_tensor)

        self.train_size = train_y.shape[0]
        self.test_size = test_y.shape[0]

    def build_net(self):
        """
        builds net from scratch. If you want to load a custom net, use self.load_net
        """
        self.net = nets.PurchaseNet(input_dim=self.input_dim, output_dim=self.n_labels)


class CifarModel(Model):
    def load_data(self, train_size=50000, test_size=10000, p_noise=0.0, augment=False):
        print(f"Loading CIFAR10 from torchvision: {train_size} train, {test_size} test.")
        
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        # add noise to training labels
        if (p_noise > 0):
            ind = np.random.choice(len(trainset.targets), int(p_noise*len(trainset.targets)), replace=False)
            for i in ind:
                trainset.targets[i] = np.random.randint(10)
        
        # subset datasets
        self.trainset = random_split(trainset, [train_size, trainset.data.shape[0] - train_size])[0]
        self.testset = random_split(testset, [test_size, testset.data.shape[0] - test_size])[0]

        self.train_size = train_size
        self.test_size = test_size

    def build_net(self, k=64, num_classes=10):
        self.net = nets.make_resnet18k(k, num_classes)

    def _auc(self, set_1, set_2, big = False):
        with torch.no_grad():

            if big:
                train_confidence = np.zeros(set_1.shape[0])
                test_confidence = np.zeros(set_2.shape[0])

                idx = 0
                batch_size = 500
                num_batches = set_1.shape[0] // batch_size
                for batch in np.split(set_1, num_batches):
                    train_confidence_here = torch.max(torch.nn.Softmax(dim=-1)(self.net(torch.Tensor(batch).to(self.device)).detach()), axis=1)[0]
                    train_confidence[idx: idx + batch.shape[0]] = train_confidence_here.cpu().numpy()
                    idx += batch.shape[0]
                train_confidence = torch.tensor(train_confidence)

                idx = 0
                batch_size = 500
                num_batches = set_2.shape[0] // batch_size
                for batch in np.split(set_2, num_batches):
                    test_confidence_here = torch.max(torch.nn.Softmax(dim=-1)(self.net(torch.Tensor(batch).to(self.device)).detach()), axis=1)[0]
                    test_confidence[idx: idx + batch.shape[0]] = test_confidence_here.cpu().numpy()
                    idx += batch.shape[0]
                test_confidence = torch.tensor(test_confidence)
            else:
                train_confidence = torch.max(torch.nn.Softmax(dim=-1)(self.net(torch.Tensor(set_1).to(self.device)).detach()), axis=1)[0]
                test_confidence = torch.max(torch.nn.Softmax(dim=-1)(self.net(torch.Tensor(set_2).to(self.device)).detach()), axis=1)[0]

            actual = [1 for i in range(set_1.shape[0])] + [0 for i in range(set_2.shape[0])]
            estimated = torch.cat([train_confidence, test_confidence])
            return roc_auc_score(torch.Tensor(actual), estimated.cpu())