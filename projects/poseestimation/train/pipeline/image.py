import torchvision.models as models

model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
model = model.cuda()
model.eval()

