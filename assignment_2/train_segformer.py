
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import torch.optim as optim
from pathlib import Path


from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer

# MODIFY IF NEEDED
path_oxford = "/caa/Student/dlvc/dlvc11711533"
path_cityscape = "/caa/Student/dlvc/dlvc11711533/cityscapes_assg2"

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

    if args.dataset == "oxford":
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
    if args.dataset == "city":
        train_data = CityscapesCustom(root=path_cityscape, 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root=path_cityscape, 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # number of classes and batch size depending on the dataset
    if args.dataset == 'oxford':
        print("Training on Oxford Pets dataset")
        num_classes = 3
        batch_size = 64
    else:
        print("Training on Cityscapes dataset")
        num_classes = 19
        batch_size = 16

    model = DeepSegmenter(SegFormer(num_classes=num_classes))

    # If you are in the fine-tuning phase:
    ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
    # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')

    # use pretrained weights
    if args.finetune:
        print("Finetuning is enabled")
        #lower the learning rate for finetuning
        learning_rate = 0.0005
        #learning_rate = 0.001
        # load state_dict from pretrained model (including encoder and decoder weights)
        state_dict = torch.load(args.pretrained_weights_path, map_location='cpu')
        # loop over the state dict: remove the encoder prefix from the keys and exclude the decoder weights entirely
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "", 1)
                encoder_state_dict[new_key] = value
        # load the encoder weights into the model
        model.net.encoder.load_state_dict(encoder_state_dict)

        # print model weights - for debugging
        print("Pretrained model weights")
        for param in model.parameters():
            print(param.data)

        # freeze the encoder weights
        if args.freeze_encoder:
            print("Freezing the encoder weights")
            for param in model.net.encoder.parameters():
                param.requires_grad = False
            # or: model.net.encoder.requires_grad_(False)
    else:
        learning_rate = 0.001
        # print model weights - for debugging
        print("Model weights without pretraining")
        for param in model.parameters():
            print(param.data)

    model.to(device)

    # if the encoder is frozen, only the decoder parameters are optimized
    if args.freeze_encoder:
        optimizer = optim.Adam(model.net.decoder.parameters(), lr=learning_rate, amsgrad=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    
    if args.dataset == "city":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 255) # remember to ignore label value 255 when training with the Cityscapes datset
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 

    model_save_dir = Path("saved_models" + args.save_subdir)
    model_save_dir.mkdir(exist_ok=True)

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
                    batch_size=batch_size,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    args.add_argument('--num_epochs', default = 30, type = int, help = 'number of epochs for training'),
    args.add_argument('--dataset', choices = ['oxford', 'city'], type = str, required = True, help = 'dataset to train on')
    args.add_argument('--save_subdir', default = '', type = str, help = 'name of the subdirectory to save the model to') # specify a separate subdirectory for the model to be saved to, if wanted
    args.add_argument('--finetune', action = 'store_true', help = 'whether to finetune the model') # evaluates to False if not set
    args.add_argument('--pretrained_weights_path', default = '', type = str, help = 'path to the pretrained encoder weights')
    args.add_argument('--freeze_encoder', action = 'store_true',  help = 'whether to freeze the encoder weights') # evaluates to False if not set

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    train(args)