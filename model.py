from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


# Function to load deeplabv3 pretrained and change outputchannel of its classifier head to number of classes
def createDeepLabHead(outputchannels= 1):
    """ Custom Deeplab Classifier head """

    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels) # Adjust classifier head, resnet101 has backbone output of 2048
    model.aux_classifier = FCNHead(1024, outputchannels) # Adjust aux classifier
    return model
