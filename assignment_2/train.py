#!/bin/python3
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
import torch.optim as optim
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from torchvision.models.segmentation.fcn import ResNet50_Weights

#Path for server 
path_oxford = "/caa/Student/dlvc/dlvc11711533"

def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root=path_oxford, 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root=path_oxford, 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Task 4.1
    #model = DeepSegmenter(fcn_resnet50(pretrained=False, progress=True, num_classes=3))
    #Task 4.2
    #model = DeepSegmenter(fcn_resnet50(pretrained=True, progress=True, num_classes=3, weights_backbone=ResNet50_Weights.IMAGENET1K_V1))
    #model = DeepSegmenter(fcn_resnet50(pretrained=True, progress=True, num_classes=3, weights_backbone=ResNet50_Weights.IMAGENET1K_V1_FREEZE))
    #model = DeepSegmenter(fcn_resnet50(pretrained=True, progress=True, num_classes=3))
    #model = DeepSegmenter(fcn_resnet50(progress=True, num_classes=3, weights_backbone=ResNet50_Weights.DEFAULT))
    model = DeepSegmenter(fcn_resnet50(progress=True, num_classes=3, weights_backbone=ResNet50_Weights.IMAGENET1K_V2))

    # Adamp optimizer with amsgrad and lr=0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    # ExponentialLR scheduler with gamma= 0.98
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument('--num_epochs', default = 30, type = int,
                      help = 'number of epochs for training')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)