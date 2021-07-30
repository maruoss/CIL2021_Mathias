from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


# Function to load deeplabv3 pretrained and change outputchannel of its classifier head to number of classes
def createDeepLabHead_mobilenet(outputchannels= 1):
    """ Custom Deeplab Classifier head """

    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
    model.classifier = DeepLabHead(960, outputchannels) # Adjust classifier head, resnet101 has backbone output of 2048
    model.aux_classifier = FCNHead(40, outputchannels) # Adjust aux classifier
    return model
