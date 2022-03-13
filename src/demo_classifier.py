import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data

import torchvision.transforms as T
from utilities  import *
import numpy as np
from builtins import *
from torchsummary import summary



class Demo_Classifier:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    def preprocessing(self):
        #transform = T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        #data_set = torchvision.datasets.ImageFolder(root='../data', transform=transform)
        X,y= import_data()
        data_set=[]
        for i in range(len(X)):         
            data_set.append([X[i],y[i]])
        train_set_size = int(len(data_set) *0.8)
        test_set_size = len(data_set) - train_set_size
        train_set, test_set = data.random_split(data_set, [train_set_size, test_set_size])
        loader_train = DataLoader(train_set, batch_size=64)
        loader_test = DataLoader(test_set, batch_size=64)
        return loader_train, loader_test
        
    def cnn_model(self):
        layer1 = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )

        layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )


        fc = nn.Linear(32*5*4, 4)

        model = nn.Sequential(
        layer1,
        layer2,
        nn.Flatten(),
        fc
        )
        return model

    def check_accuracy(self,loader,model):
        num_correct = 0
        num_samples = 0
        model.eval()  
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=self.dtype) 
                y = y.to(device=self.device, dtype=torch.int64)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
    def train_part(self,model,loader_train):
        epochs = 5
        learning_rate = 0.001
        weight_decay = 0.01
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model = model.to(device=self.device) 
        for e in range(epochs):
            for t,(x, y) in enumerate(loader_train):
                model.train()  
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=torch.int64)
                scores = model(x)
                loss = torch.nn.CrossEntropyLoss()(scores, y)

           
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                if t % 100 == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self.check_accuracy(loader_train, model)
                    print()
                    
        return model
            
    
        
loader_train_set,loader_test_set = Demo_Classifier().preprocessing()   
model = Demo_Classifier().cnn_model()
summary(model,(4,210,160))
print('Training Set Accuracy')
model_aftertrain=Demo_Classifier().train_part(model,loader_train_set)
print('Test Set Accuracy:')
Demo_Classifier().check_accuracy(loader_test_set, model_aftertrain)   
    
    
    
    
    
    