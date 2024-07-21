from torchvision.models import resnet50, ResNet50_Weights


# based on the tutorial from pytorch: Models and pre-trained weights
# source: https://pytorch.org/vision/stable/models.html

RN50_Weights = [ResNet50_Weights.IMAGENET1K_V1,
                ResNet50_Weights.IMAGENET1K_V2, ResNet50_Weights.DEFAULT, None]


def weights(model_weights: ResNet50_Weights):

    model = resnet50(weights=model_weights)

    for k, v in model.state_dict().items():
        print(k, v)
        return


weights(RN50_Weights[0])
