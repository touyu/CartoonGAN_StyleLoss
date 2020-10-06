import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, utils

def get_features(image, model):
    """
    note : CNNの特定層から特徴量を抽出する関数
    ----------
    image : tensor
    model : type of CNN(今回はVGG19)
    ----------
    """
    layers = {
        "0" : "conv1_1",
        "5" : "conv2_1",
        "10" : "conv3_1",
        "19" : "conv4_1",
        "28" : "conv5_1",
    }

    features = {}

    for name, layer in model._modules.items():
        print(name)

    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image

    return features


vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

for name, layer in vgg._modules.items():
    print(name, layer)
