import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models

# ===========================================
#            Digit recognition
# ===========================================

class LeNet(nn.Module):
    def __init__(self, n_classes=10, target=False):
        super(LeNet, self).__init__()
        
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(n_classes=n_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.feature1 = nn.Sequential(
                    nn.Conv2d(1,20,kernel_size=5, stride=1, padding='valid'),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                ) 

        self.feature2 = nn.Sequential(
                    nn.Conv2d(20,50,kernel_size=5, stride=1, padding='valid'),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
        self.feature3 = nn.Linear(50 * 4 * 4, 500)                
                
    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        flatten = torch.flatten(x, 1)
        features = self.feature3(flatten)
   
        return features


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
       
        self.n_classes = n_classes
        self.classifier = nn.Linear(500, self.n_classes)
                         
    def forward(self, x):
        out = self.classifier(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,1),
#             nn.LogSoftmax() # sigmoid?
        )
        
    def forward(self, x):
        out = self.layer(x)
        return out

class Weight(nn.Module):
    def __init__(self, normalized=False, relu=False):
        super(Weight, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,1),
#             nn.ReLU()
        )
        self.normalized = normalized
        self.relu = relu

        
    def forward(self, x):
        x = self.layer(x)
        if self.normalized:
            x = x.squeeze()
            x = F.softmax(x)
            x = x * len(x)
            x = x.reshape(-1,1)
        if self.relu:
            x = F.relu(x)
        else:
            x = x
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100,1),
            )
        
    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred
    
# ===========================================
#            Office-31
# ===========================================
    
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        
        self.freezed = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.in_features_ = resnet.fc.in_features              
                
    def forward(self, x):
        x = self.freezed(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        res_features = x.view(x.size(0), -1)
        return res_features

class FeatureExtractor2(nn.Module):
    def __init__(self, in_channels=3, h=256, dropout=0.5, freeze_first=False):
        super(FeatureExtractor2, self).__init__()
        
        rn_model = ResNetModel()
        self.freezed = rn_model.freezed
        self.layer4 = rn_model.layer4
        self.avgpool = rn_model.avgpool
        if freeze_first:
            for param in self.freezed.parameters():
                param.requires_grad = False #??? deze freezed heb je toch false niet alleen sourc
        
    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 224, 224)
        x = self.freezed(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        return features

class Classifier2(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier2, self).__init__()
        self.classifier = nn.Linear(2048, n_classes)
                         
    def forward(self, x):
        out = self.classifier(x)
        return out
    
class CNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=31, target=False, freeze_first=False):
        super(CNN, self).__init__()
        self.feature_extractor = FeatureExtractor2(in_channels=in_channels,freeze_first=freeze_first)
        self.classifier = Classifier2(n_classes=n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(2048, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500,1)
        )
        
    def forward(self, x):
        out = self.layer(x)
        return out
    

class Discriminator2b(nn.Module):
    def __init__(self):
        super(Discriminator2b, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(2048, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500,2),
            nn.LogSoftmax(dim=1) 
        )
        
    def forward(self, x):
        out = self.layer(x)
        return out
    
    
# ===========================================
#            Office DECAF
# ===========================================


class LeNetDC(nn.Module):
    def __init__(self, n_classes=10, target=False):
        super(LeNetDC, self).__init__()
        
        self.feature_extractor = FeatureExtractorDC()
        self.classifier = ClassifierDC(n_classes=n_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class LeNetDC2(nn.Module):
    def __init__(self, n_classes=10, target=False):
        super(LeNetDC, self).__init__()
        
        self.feature_extractor = FeatureExtractorDC2()
        self.classifier = ClassifierDC(n_classes=n_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class FeatureExtractorDC(nn.Module):
    def __init__(self):
        super(FeatureExtractorDC, self).__init__()

        self.block = nn.Sequential(
                    nn.Linear(4096, 500),
                    nn.ReLU(),
                    nn.Linear(500, 100),
                    nn.ReLU() 
                        )
                
    def forward(self, x):
        out = self.block(x)
        return out

class FeatureExtractorDC2(nn.Module):
    def __init__(self):
        super(FeatureExtractorDC2, self).__init__()

        self.block = nn.Sequential(
                    nn.Linear(4096, 500),
                    nn.LeakyReLU(),
                    nn.Linear(500, 100),
                    nn.LeakyReLU(),
                    nn.Linear(100, 100)
                        )
                
    def forward(self, x):
        out = self.block(x)
        return out

class ClassifierDC(nn.Module):
    def __init__(self, n_classes):
        super(ClassifierDC, self).__init__()
        
        self.n_classes = n_classes
        self.block = nn.Linear(100, self.n_classes)
            
    def forward(self, x):
        out = self.block(x)
        return out  

class DiscriminatorDC(nn.Module):
    def __init__(self):
        super(DiscriminatorDC, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,1),
        )

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred

class DiscriminatorDC2(nn.Module):
    def __init__(self):
        super(DiscriminatorDC2, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100,1))

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred
    
    

# class LeNet(nn.Module):
#     def __init__(self, n_classes):
#         super(LeNet, self).__init__()
       
#         self.n_classes = n_classes
        
#         self.feature1 = nn.Sequential(
#                     nn.Conv2d(1,20,kernel_size=5, stride=1, padding='valid'),
#                     nn.ReLU(),
#                     nn.MaxPool2d(kernel_size=2,stride=2)
#                 ) 

#         self.feature2 = nn.Sequential(
#                     nn.Conv2d(20,50,kernel_size=5, stride=1, padding='valid'),
#                     nn.ReLU(),
#                     nn.MaxPool2d(kernel_size=2, stride=2),
#                 )
        
#         self.feature3 = nn.Linear(50 * 4 * 4, 500)
        
#         self.classifier = nn.Sequential(nn.Linear(500, self.n_classes),
# #                     nn.Softmax(dim=1)
#                 )
                
                
#     def forward(self, x):
#         x = self.feature1(x)
#         x = self.feature2(x)
#         flatten = torch.flatten(x, 1)
#         features = self.feature3(flatten)
#         out = self.classifier(features)
   
#         return out, features
