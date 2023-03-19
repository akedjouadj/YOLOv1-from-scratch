import torch
import torch.nn as nn

""" 
YOLOv1 architecture config: conv layers
Tuple structure: (kernel_size, filters, stride, padding). The padding is calculated by hand
"M" for maxpooling: stride 2x2, kernel 2x2
List structure: tuples and int, the number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module): # a general CNN block class that we'll often use
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias= False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): # in_channels : 3 for RGB
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) # build from architecture
        self.fcs = self._create_fcs(**kwargs) # fcs for fully connected layers

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture): # create the darknet architecture
        layers= []
        in_channels = self.in_channels

        for layer in architecture:
            if type(layer) == tuple:
                layers+= [CNNBlock(in_channels=in_channels, out_channels=layer[1], kernel_size= layer[0],
                                  stride = layer[2], padding = layer[3]) ]
                in_channels = layer[1]
            
            elif type(layer) == str:
                layers+= [nn.MaxPool2d(kernel_size=2, stride = 2)]
            
            elif type(layer) == list:
                conv1 = layer[0] # tuple
                conv2 = layer[1] # tuple
                num_repeats = layer[2]

                for _ in range(num_repeats):
                    layers+= [CNNBlock(in_channels=in_channels, out_channels=conv1[1], kernel_size= conv1[0],
                                  stride = conv1[2], padding = conv1[3]) ]
                    layers+= [CNNBlock(in_channels=conv1[1], out_channels=conv2[1], kernel_size= conv2[0],
                                  stride = conv2[2], padding = conv2[3]) ]
                    
                in_channels = conv2[1]

        return nn.Sequential(*layers) # unpack the list layers and convert it to nn.Sequential
    
    def _create_fcs(self, split_size, num_boxes, num_classes): # create the fully connected layers architecture
        S, B, C= split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496), # 4096 in the original paper, but need to decrease that on a small computer beacause of small VRAM
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5)) # reshape by (S, S, C+B*5)
        )

# quickly test of the Yolov1 model
"""   
def test(S=7, B=2, C=20):
    model = Yolov1(split_size = S, num_boxes = B, num_classes = C)
    x = torch.randn([2, 3, 448, 448])
    print(model(x).shape)

test()
"""

             


