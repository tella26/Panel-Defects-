# Copyright (C) 2018 Sergiu Deitsch
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torchvision
from augmentations import augmentation
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os




class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, fname= ''):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.fname = fname

    def load_dataset(self, fname=None, transform=False):
            if fname is None:
                # Assume we are in the utils folder and get the absolute path to the
                # parent directory.
                fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    os.path.pardir))
                fname = os.path.join(fname, 'labels.csv')

            data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
                                'path', 'probability', 'type'])
            image_fnames = np.char.decode(data['path'])
            probs = data['probability']
            types = np.char.decode(data['type'])

            def load_cell_image(fname):
                with Image.open(fname) as image:
                    return image

            dir = os.path.dirname(fname)

            images = [load_cell_image(os.path.join(dir, fn))
                            for fn in image_fnames]
                        
            test_path = os.mkdir('test_data')
            train_path = os.mkdir('train_data')
            
        
            
            if transform:
                 transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                           transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataloader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root=test_path,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])),
                batch_size=self.batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root=train_path,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])),
                batch_size=self.batch_size, shuffle=True)

            return train_dataloader, test_dataloader

          

if __name__ == '__main__':
    # Download the data set from URL
    print("Downloading data from {}".format('../elpv_dataset'))
    train_dataloader, test_dataloader = initialize_dataset.load_dataset()
    
    print(train_dataloader.size())
    print(test_dataloader.size())


        
        
  

