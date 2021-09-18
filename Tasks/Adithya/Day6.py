import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

transforms_resnet_vgg = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
])
transforms_inception = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_set = datasets.ImageFolder("./data/data/train",transforms_resnet_vgg)
val_set = datasets.ImageFolder("./data/data/val",transforms_resnet_vgg)


train_set_inception = datasets.ImageFolder("./data/data/train",transforms_inception)
val_set_inception = datasets.ImageFolder("./data/data/val",transforms_inception)


train_loader = torch.utils.data.DataLoader(train_set,batch_size=4,shuffle=True,num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=4,shuffle=True,num_workers=4)
classes = train_set.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader_inception = torch.utils.data.DataLoader(train_set_inception,batch_size=4,shuffle=True,num_workers=4)
val_loader_inception = torch.utils.data.DataLoader(val_set_inception,batch_size=4,shuffle=True,num_workers=4)

MODEL = {
    "inceptionv3" : models.inception_v3(pretrained=True),
    "resnet50" : models.resnet50(pretrained=True),
    "vgg16" : models.vgg16(pretrained=True)
}

for name,model in MODEL.items():
    for param in model.parameters():
        param.requires_grad = False

num_features = MODEL["resnet50"].fc.in_features
MODEL["resnet50"].fc = nn.Linear(num_features,2)
MODEL["resnet50"] = MODEL["resnet50"].to(device)


num_features = MODEL["vgg16"].classifier[6].in_features
features = list(MODEL["vgg16"].classifier.children())[:-1]
features.extend([nn.Linear(num_features,2)])
MODEL["vgg16"].classifier = nn.Sequential(*features)
MODEL["vgg16"] = MODEL["vgg16"].to(device)

num_features = MODEL["inceptionv3"].fc.in_features
MODEL["inceptionv3"].fc = nn.Linear(num_features,2)
MODEL["inceptionv3"] = MODEL["inceptionv3"].to(device)
MODEL["inceptionv3"].aux_logits=False

criterion = nn.CrossEntropyLoss()
optimizer = {}
for name,model in MODEL.items():
    optimizer[name] = optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9)

for name, model in MODEL.items():
    model.train()
    for epochs in range(10):
        running_loss = 0.0
        train_loader = train_loader_inception if name == "inceptionv3" else train_loader
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer[name].zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer[name].step()

            running_loss += loss.item()
        print("loss [" + name +"] :",running_loss)
    print("Finished Training ["+ name +"]")


class_correct = [0.0,0.0]
class_total = [0.0,0.0]
for name,model in MODEL.items():
    with torch.no_grad():
        val_loader = val_loader_inception if name == "inceptionv3" else val_loader
        for i, data in enumerate(val_loader,0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs,1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print(name)
    for i in range(2):
        print('accuracy of %5s : %2d %%'%(classes[i],100*class_correct[i]/class_total[i]))


