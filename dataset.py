import torch
import torchvision
from augmentations import augmentation
import torchvision.transforms as transforms

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=25, test_path= '', train_path=''):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.train_path = train_path
        self.test_path = test_path
  
    def load_dataset(self, transform=False):
        path = "../data"
        if transform:
            transform = augmentation(image_resolution=self.image_resolution)
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                       transforms.RandomHorizontalFlip(),transforms.RandomRotation(20),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataloader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root=self.test_path,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])),
                batch_size=self.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root=self.train_path,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])),
                batch_size=self.batch_size, shuffle=True)

        return train_dataloader, test_dataloader
