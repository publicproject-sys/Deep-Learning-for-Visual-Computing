## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import torchvision
from pathlib import Path
import torch.optim as optim

# Deep learning model and related imports
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.cnn import YourCNN, YourCNNWithDropout


def train(args):
    # Define data transforms for training and validation
    if args.augmentation:
        print("Image augmentation is enabled")
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p = 0.5),  # horizontally flip the input with probability p
            v2.RandomResizedCrop(size = 32), # crop the given image
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 datasets with the correct file paths

    #Path for Vladimir
    #cifar10_dir = "C:\\Users\\awtfh\\Documents\\Programming\\Deep_VC\\cifar-10-python\\cifar-10-batches-py"
    #Path for Philipp
    #cifar10_dir = "C:\\Users\\Home\\Documents\\TU Wien\\SS 2024\\Deep Learning for Visual Computing\\dlvc_ss24\\assignments\\cifar-10-batches-py"
    #Path on cluster
    cifar10_dir = Path('/caa/Student/dlvc/dlvc11711533/cifar-10-batches-py')
    train_data = CIFAR10Dataset(
        fdir=cifar10_dir,
        subset=Subset.TRAINING,
        transform=train_transform
    )

    val_data = CIFAR10Dataset(
        fdir=cifar10_dir,
        subset=Subset.VALIDATION,
        transform=val_transform
    )

    # Determine device for training
    print("Has GPU:", torch.cuda.is_available()) # check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CNN with dropout if enabled
    if args.dropout:
        print("Dropout regularization is enabled")
        model = DeepClassifier(YourCNNWithDropout())
    else:
        model = DeepClassifier(YourCNN())

    model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Define metrics and validation frequency
    train_metric = Accuracy(classes=len(train_data.classes))
    val_metric = Accuracy(classes=len(val_data.classes))
    val_frequency = 5  

    # Create a directory for saving the model
    model_save_dir = Path("/caa/Student/dlvc/dlvc11711533/assignment_1/saved_models/yourcnn")
    #model_save_dir = Path("saved_models/yourcnn")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Create the trainer and initiate training
    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=128,
        val_frequency=val_frequency
    )
    
    # Start the training process
    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default = '0', type=str,
                      help = 'index of which GPU to use')
    args.add_argument('--num_epochs', default = 25, type = int,
                      help = 'number of epochs for training')
    args.add_argument('--augmentation', default = False, type = bool,
                      help = 'True or False flag for data augmentation')
    args.add_argument('--dropout', default = False, type = bool,
                      help = 'True or False flag for dropout')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0 
    args.num_epochs = 25
    args.augmentation = False
    args.dropout = False

    train(args)