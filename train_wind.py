
from os import TMP_MAX
import torch
import torch.nn as nn
import numpy as np
from optimizer import optim 
from pathlib import Path
from plot import trainTestPlot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Training:
    
    def __init__(self, model, optimizer, learning_rate, train_dataloader, num_epochs, 
                test_dataloader, eval=True, plot=False, model_name=None, model_save=False):
        self.model = model
        self.learning_rate = learning_rate
        self.optim = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.eval = eval
        self.plot = plot
        self.model_name = model_name
        self.model_save = model_save

    def runner(self):
        criterion = nn.CrossEntropyLoss()   
        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)      
        else:
            pass
        
      # loop over the dataset multiple times
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print('Loss: {}'.format(running_loss))

        print('Finished Training')
