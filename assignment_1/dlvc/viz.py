import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np

import os
print(os.getcwd())


from datasets.cifar10 import CIFAR10Dataset
from datasets.dataset import Subset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png",np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__': 
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True)])


    #cifar10_dir = "C:\\Users\\awtfh\\Documents\\Programming\\Deep_VC\\cifar-10-python\\cifar-10-batches-py"
    cifar10_dir = "C:\\Users\\Home\\Documents\\TU Wien\\SS 2024\\Deep Learning for Visual Computing\\dlvc_ss24\\assignments\\cifar-10-batches-py"
    train_data = CIFAR10Dataset(fdir=cifar10_dir, subset=Subset.TRAINING, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=2)

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
