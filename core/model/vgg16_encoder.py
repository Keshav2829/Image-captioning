import torch as T
from torch import nn

# input image size -> 224*224*3
class ConvBlock2layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding="same")
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding="same")
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, input):
        x = T.relu(self.conv1(input))
        x = self.maxpool(T.relu(self.conv2(x)))
        return x


class ConvBlock3layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding="same")
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding="same")
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, input):
        x = T.relu(self.conv1(input))
        x = T.relu(self.conv2(x))
        x = self.maxpool(T.relu(self.conv3(x)))
        return x
    

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ConvBlock2layer(3,64) # input-> 3*224*224 output-> 64*112*112
        self.layer2 = ConvBlock2layer(64,128) # input-> 64*112*112 output-> 128*56*56
        self.layer3 = ConvBlock3layer(128,256) # input-> 128*56*56 output-> 256*28*28
        self.layer4 = ConvBlock3layer(256,512) # input-> 256*28*28 output-> 512*14*14
        self.layer5 = ConvBlock3layer(512,512) # input-> 512*14*14 output-> 512*7*7

        self.flatten = nn.Flatten() # input-> 512*7*7 output-> 1*25088
        self.fc1 = nn.Linear(25088, 4096) #input-> 25088 output-> 4096
        self.fc2 = nn.Linear(4096, 1000)

        self.softmax = nn.Softmax(dim=-1) # last layer -> output -> 1*1000
    
    def forward(self, input):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(input)))))
        x = self.flatten(x)
        x = self.softmax(self.fc2(T.relu(self.fc1(x))))
        return x




class VGGHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input 512,7,7 ->512,4,4-> 4*4*512 -> embed_dim
        self.avgpool = nn.AdaptiveMaxPool2d((4,4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*4*4, config['img_embed_dim'])
        self.batchnorm = nn.BatchNorm1d(config['img_embed_dim'])
        self.activation = nn.ReLU()

    def forward(self, input):
       # input shape b,c,w,h
       x = self.flatten(self.avgpool(input))
       x = self.batchnorm(self.activation(self.fc(x)))
    #    batchsize = input.shape[0]
    #    channels = input.shape[1]
    #    input = input.permute(0,2,3,1).view(batchsize,-1, channels).mean(dim=1)
       return x