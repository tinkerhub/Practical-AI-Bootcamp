import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pickle
import io

# Model Architecture
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()

        # Convolutional layers
        self.conv1= nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        self.conv3= nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5)
        self.conv4= nn.Conv2d(in_channels=24,out_channels=48,kernel_size=5)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(in_features=48*12*12,out_features=240)
        self.fc2 = nn.Linear(in_features=240,out_features=120)
        self.out = nn.Linear(in_features=120,out_features=2)
        
        
    def forward(self,t):
        t = t
        
        t=self.conv1(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv3(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv4(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        t=t.reshape(-1,48*12*12)
        t=self.fc1(t)
        t=F.relu(t)
        t=self.fc2(t)
        t=F.relu(t)
        
        t=self.out(t)
        
        return t


@torch.no_grad()
def predict(model,imgdata):
    with open('model_scripts/labels.json', 'rb') as lb:
        labels = pickle.load(lb)

    # Load the trained model
    loaded_model = model
    loaded_model.load_state_dict(torch.load("model_scripts/model.pth"))
    loaded_model.eval()

    # Converting Base64 string to Image
    image = Image.open(io.BytesIO(imgdata))
    # Resizing Image
    resize = transforms.Compose([transforms.Resize((256,256))])
    image = ToTensor()(image)

    y_result = model(resize(image).unsqueeze(0))
    res_index = y_result.argmax(dim=1)
    res = ""

    for key,value in labels.items():
        if(value==res_index):
            res = key
            break

    return res 