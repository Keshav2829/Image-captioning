import torchvision
import torch

model = torchvision.models.vgg16()
print(model)
# torch.save(model.state_dict(), r'C:\local_data\projects\image_captioning\data\weights\VGG_16\weights.pth')