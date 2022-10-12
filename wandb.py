import os
import yaml
import wandb
import argparse
import torch

import torch.nn as nn
from dataset import initialize_dataset

from convNet import CNN
from AlexNet import AlexNet
from DenseNet import DenseNet
from GoogLeNet import GoogLeNet
from ResNet import ResNet
from ResNet34 import ResNet34
from ResNet50 import ResNet50, Bottleneck
from ResNet101 import ResNet101,  Bottleneck
from ResNet152 import ResNet152,  Bottleneck
from SENet import SENet
from VGG import VGG11
from NiN import NIN
from MLPMixer import MLPMixer
from MobileNetV1 import MobileNetV1
from InceptionV3 import InceptionV3
from Xception import Xception
from ResNeXt import ResNeXt29_2x64d
from ViT import ViT
from MobileNetV2 import MobileNetV2
from Darknet53 import Darknet53
from SqueezeNet import SqueezeNet
from ShuffleNet import ShuffleNet
from EfficientNet import EfficientNet
from ResMLP import ResMLP


class Model:
    def __init__(self, input_channel, n_classes, image_resolution, model):
        self.input_channel = input_channel
        self.image_resolution = image_resolution
        self.n_classes = n_classes
        self.model = model

    def model_selection(self):

        """Model Initialization"""

        if self.model == 'vggnet':
            model = VGG11(input_channel=self.input_channel, n_classes=self.n_classes,
                    image_resolution=self.image_resolution).to(device)

        elif self.model == 'alexnet':
            model = AlexNet(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'senet':
            model = SENet(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'resnet18':
            model = ResNet(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'resnet34':
                model = ResNet34(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'resnet50':
                model = ResNet50( Bottleneck, num_channels=self.input_channel, num_classes=self.n_classes).to(device)
                
        elif self.model == 'resnet101':
                model = ResNet101(Bottleneck, num_channels=self.input_channel, num_classes=self.n_classes).to(device)
                
        elif self.model == 'resnet152':
                model = ResNet152(Bottleneck, num_channels=self.input_channel, num_classes=self.n_classes).to(device)

        elif self.model == 'densenet':
            model = DenseNet(input_channel=self.input_channel, n_classes=self.n_classes, 
                    growthRate=12, depth=100, reduction=0.5, bottleneck=True).to(device)

        elif self.model == 'nin':
            model = NIN(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'googlenet':
            model = GoogLeNet(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'cnn':
            model = CNN(input_channel=self.input_channel).to(device)

        elif self.model == 'mobilenetv1':
            model = MobileNetV1(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'inceptionv3':
            model = InceptionV3(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'xception':
            model = Xception(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'resnext':
            model = ResNeXt29_2x64d(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'vit':
            model = ViT(image_size=self.image_resolution, patch_size=32, dim=1024, depth=6, heads=16, 
                    input_channel=self.input_channel, n_classes=self.n_classes,  mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)

        elif self.model == 'mobilenetv2':
            model = MobileNetV2(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'darknet':
            model = Darknet53(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'squeezenet':
            model = SqueezeNet(input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model == 'shufflenet':
            cfg = {'out': [200,400,800], 'n_blocks': [4,8,4], 'groups': 2}
            model = ShuffleNet(cfg=cfg, input_channel=self.input_channel, n_classes=self.n_classes).to(device)

        elif self.model in ['efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 
                            'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']:
            param = {
                # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
                'efficientnetb0': (1.0, 1.0, 224, 0.2), 'efficientnetb1': (1.0, 1.1, 240, 0.2),
                'efficientnetb2': (1.1, 1.2, 260, 0.3), 'efficientnetb3': (1.2, 1.4, 300, 0.3),
                'efficientnetb4': (1.4, 1.8, 380, 0.4), 'efficientnetb5': (1.6, 2.2, 456, 0.4),
                'efficientnetb6': (1.8, 2.6, 528, 0.5), 'efficientnetb7': (2.0, 3.1, 600, 0.5)
            }
            model = EfficientNet(input_channels=self.input_channel, param=param[args.model], n_classes=self.n_classes).to(device)

        elif self.model == 'mlpmixer':
            model = MLPMixer(image_size =self.image_resolution, input_channels = self.input_channel,
            patch_size = 16, dim = 512, depth = 12, n_classes =self.n_classes, token_dim=128, channel_dim=1024).to(device)

        elif self.model == 'resmlp':
            model = ResMLP(in_channels=self.input_channel, image_size=self.image_resolution, patch_size=16, 
                    n_classes=self.n_classes, dim=384, depth=12, mlp_dim=384*4).to(device)

        return model

def train():
    # Default Config
    config_defaults = {
            'num_epochs': 3,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'image_resolution': 224,
            'MNIST': False,
            'n_classes': 12,
            'train_path': '/content/drive/MyDrive/dataset/train_data',
            'test_path' : '/content/drive/MyDrive/dataset/test_data'
        }    
    
    wandb.init(config=config_defaults)
    config = wandb.config

    """Dataset Initialization"""
    data_initialization = initialize_dataset(image_resolution=config.image_resolution, batch_size=config.batch_size, 
                        MNIST=config.MNIST, test_path=config.test_path,  train_path=config.train_path)
    train_dataloader, test_dataloader = data_initialization.load_dataset()

    input_channel = next(iter(train_dataloader))[0].shape[1]
    n_classes = config['n_classes']

    model_object = Model(input_channel=input_channel, n_classes=n_classes, image_resolution=config.image_resolution, model="resnet18")
    classifier = model_object.model_selection()

    criterion = nn.CrossEntropyLoss()
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate)
    else:
        pass

    # Train the model
    total_step = len(train_dataloader)
    for epoch in range(config.num_epochs):
        closs = 0
        
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            closs = closs + loss.item()
            optimizer.step()
            wandb.log({"batch_loss": loss.item()})
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, config.num_epochs, i+1, total_step, loss.item()))
                # Loss.append(loss.cpu().detach().numpy())
                # visual.plot_loss(np.mean(Loss), i)
                # Loss.clear()
        wandb.log({"loss":closs/config.batch_size})         

    if eval:
        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy: {(correct*100)/total}')


"""Device Selection"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters':{
    
      'num_epochs': {
          'values': [2, 5, 10]
      },
      'batch_size': {
          'values': [128, 64, 32]
      },
      'learning_rate': {
          'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
      },
      'optimizer': {
          'values': ['adam', 'sgd']
      },
      'image_resolution': {
          'values': [96, 128, 224, 256]
      }
    }
}

sweep_id = wandb.sweep(config, project="Solar Panel Defect Detection")