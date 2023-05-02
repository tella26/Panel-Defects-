import argparse

import torch
import yaml
from torchstat import stat  # install torchstat

from models.convNet import CNN
from models.AlexNet import AlexNet
from models.GoogLeNet import GoogLeNet
from models.ResNet import ResNet
from models.ResNet34 import ResNet34
from models.ResNet50 import ResNet50, Bottleneck
from models.ResNet101 import ResNet101,  Bottleneck
from models.ResNet152 import ResNet152,  Bottleneck
from models.SENet import SENet
from models.Xception import Xception
from models.ViT import ViT
from models.Darknet53 import Darknet53
from models.SqueezeNet import SqueezeNet  


from dataset import initialize_dataset
from train_test import Training


"""Device Selection"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Initialize model based on command line argument """
model_parser = argparse.ArgumentParser(description='Solar panel cell Classification Using PyTorch', usage='[option] model_name')
model_parser.add_argument('--model', type=str, required=True)
model_parser.add_argument('--train_path', type=str)
model_parser.add_argument('--test_path', type=str)
model_parser.add_argument('--model_save', type=bool, required=False)
model_parser.add_argument('--checkpoint', type=bool, required=False)



args = model_parser.parse_args()

"""Loading Config File"""
try:
    stream = open("config.yaml", 'r')
    config = yaml.safe_load(stream)
except FileNotFoundError:
    print("Config file missing")


"""Dataset Initialization"""
data_initialization = initialize_dataset(image_resolution=config['parameters']['image_resolution'], batch_size=config['parameters']['batch_size'],
                       test_path=args.test_path,  train_path=args.train_path
                      #MNIST=config['parameters']['MNIST']
                      )
train_dataloader, test_dataloader = data_initialization.load_dataset(transform=True)

input_channel = next(iter(train_dataloader))[0].shape[1]
#n_classes = len(torch.unique(next(iter(train_dataloader))[1]))
n_classes = config['parameters']['n_classes']

"""Model Initialization"""

if args.model == 'alexnet':
    model = AlexNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'senet':
    model = SENet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnet18':
    model = ResNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnet34':
        model = ResNet34(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnet50':
        model = ResNet50( Bottleneck, num_channels=input_channel, num_classes=n_classes).to(device)
        
elif args.model == 'resnet101':
        model = ResNet101(Bottleneck, num_channels=input_channel, num_classes=n_classes).to(device)
        
elif args.model == 'resnet152':
        model = ResNet152(Bottleneck, num_channels=input_channel, num_classes=n_classes).to(device)

elif args.model == 'googlenet':
    model = GoogLeNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'cnn':
    model = CNN(in_channel=input_channel).to(device)

elif args.model == 'xception':
    model = Xception(input_channel=input_channel, n_classes=n_classes).to(device)
    
elif args.model == 'vit':
    model = ViT(image_size=config['parameters']['image_resolution'], patch_size=32, dim=1024, depth=6, heads=16, 
            input_channel=input_channel, n_classes=n_classes,  mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)

elif args.model == 'darknet':
    model = Darknet53(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'squeezenet':
    model = SqueezeNet(input_channel=input_channel, n_classes=n_classes).to(device)


#print(device)

print(f'Total Number of Parameters of {args.model.capitalize()} is {round((sum(p.numel() for p in model.parameters()))/1000000, 2)}M')
trainer = Training(model=model, optimizer=config['parameters']['optimizer'], learning_rate=config['parameters']['learning_rate'], 
                train_dataloader=train_dataloader, num_epochs=config['parameters']['num_epochs'],test_dataloader=test_dataloader,
                model_name=args.model, model_save=args.model_save, checkpoint=args.checkpoint)
trainer.runner()

