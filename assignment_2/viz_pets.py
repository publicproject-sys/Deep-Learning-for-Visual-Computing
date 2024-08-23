import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt
import numpy as np

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
os.chdir(os.getcwd())

from train import OxfordPetsCustom

path_oxford = "<path_to_dataset>"

def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min()) # normalize the image to 0 - 1
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__': 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the model
    num_classes = 3
    model = DeepSegmenter(SegFormer(num_classes=num_classes))

    # load the state dictionary
    pretrained_weights_path = "./saved_models/segformer_oxford_from_pretrained/SegFormer_model_best.pth"
    state_dict = torch.load(pretrained_weights_path, map_location='cpu')

    # loop over the state dict: remove the encoder prefix from the keys and exclude the decoder weights entirely
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "", 1)
            encoder_state_dict[new_key] = value
    # load the encoder weights into the model
    model.net.encoder.load_state_dict(encoder_state_dict)

    model.to(device)
    model.eval()

    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    val_data = OxfordPetsCustom(root=path_oxford, 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform,
                                download=True)
    val_data_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=8,
                                            shuffle=False)

    #val_iter = iter(val_data_loader) 
    #val_images, val_labels = next(val_iter)
    
    for i, (val_images, val_labels) in enumerate(val_data_loader):
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            val_preds = model(val_images)

        val_images_plot = torchvision.utils.make_grid(val_images, nrow=4)
        val_labels_plot = torchvision.utils.make_grid((val_labels-1)/2, nrow=4)
        val_preds_plot = torchvision.utils.make_grid((val_preds.argmax(dim=1, keepdim=True)-1)/2, nrow=4)

        # show/plot images
        imshow(val_images_plot, filename=f"img/input_test_pets_batch_{i+1}.png")
        imshow(val_labels_plot, filename=f"img/val_seg_mask_test_pets_batch_{i+1}.png")
        imshow(val_preds_plot, filename=f"img/val_pred_mask_test_pets_batch_{i+1}.png")
        
        if i == 3:
            break
