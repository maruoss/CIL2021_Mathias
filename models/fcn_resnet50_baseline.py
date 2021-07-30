from torchvision import models
from torchvision.models.segmentation.fcn import FCNHead


# Function to load FCN resnet 50 pretrained and change outputchannel of its classifier head to number of classes
def createFCNHead(outputchannels= 1):
    """ Custom Deeplab Classifier head """

    model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
    model.classifier = FCNHead(2048, outputchannels) # Adjust classifier head, resnet50 has backbone output of 2048
    model.aux_classifier = FCNHead(1024, outputchannels) # Adjust aux classifier
    return model