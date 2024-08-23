## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet18 # change to the model you want to test
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from dlvc.models.vit import VisionTransformer
from dlvc.models.cnn import YourCNN



def test(args):
    # define data transforms for testing
    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    # cifar 10 directory
    #cifar10_dir = "C:\\Users\\awtfh\\Documents\\Programming\\Deep_VC\\cifar-10-python\\cifar-10-batches-py"
    #cifar10_dir = "C:\\Users\\Home\\Documents\\TU Wien\\SS 2024\\Deep Learning for Visual Computing\\dlvc_ss24\\assignments\\cifar-10-batches-py"
    cifar10_dir = Path('/caa/Student/dlvc/dlvc11711533/cifar-10-batches-py')
    # define data loader for test set
    test_data = CIFAR10Dataset(
        fdir = cifar10_dir,
        subset = Subset.TEST,
        transform = transform
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size = 128, shuffle = False
    )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    # initialize model and load the trained weights
    if args.model_type == "resnet18":
        model = DeepClassifier(resnet18())
    elif args.model_type == "CNN":
        model = DeepClassifier(YourCNN())
    elif args.model_type == "ViT":
        model = DeepClassifier(VisionTransformer(
            embed_dim = 256,
            hidden_dim = 512,
            num_channels = 3,
            num_heads = 8,
            num_layers = 6,
            num_classes = 10,
            patch_size = 4,
            num_patches = 64,
            dropout = 0
        ))
    else:
        raise ValueError("Please define a model type")
    
    model.load(args.path_to_trained_model, args.model_type)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(test_data.classes)

    test_metric = Accuracy(classes=len(test_data.classes))

    ### Below implement testing loop and print final loss 
    ### and metrics to terminal after testing is finished
    model.eval()
    test_loss = 0.0
    test_metric.reset()

    with torch.no_grad():
        for images, labels in tqdm(test_data_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # update the loss and metric
            test_loss += loss.item()
            test_metric.update(outputs, labels)
    
    mean_loss = test_loss / len(test_data_loader)
    mean_accuracy = test_metric.accuracy()
    per_class_accuracy = test_metric.per_class_accuracy()

    print(f"Test Loss: {mean_loss:.4f}")
    print(f"Test Accuracy: {mean_accuracy:.4f}")
    for class_idx, acc in per_class_accuracy.items():
        print(f"Accuracy for class: {test_data.classes[class_idx]} is {acc:.4f}")

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='5', type=str,
                      help='index of which GPU to use')
    args.add_argument('--path_to_trained_model', default = "", type = str,
                      help = 'path to the trained model')
    args.add_argument('--model_type', default = "resnet18", type = str,
                      help = 'type of model to be used for testing')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0 
    #args.path_to_trained_model = "./saved_models/yourvit/best_model_epoch_15.pth"
    #args.path_to_trained_model = "./saved_models/resnet18_original/best_model_epoch_25.pth"
    #args.model_type = "ViT"

    test(args)