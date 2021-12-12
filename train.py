import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, USPS
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import io
import copy
import math
import seaborn as sns
import pandas as pd
import pickle
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False

import networks
import utils

root = '/opt/app-root/s3_home/'
source = 'mnist'
target = 'usps'
model_name = 'exp003_wgt'
style = 'shen'
direction = source + '_' + target
imbalanced = True
weighted = False

if imbalanced:
    dataset_path = root + '_subsets/mnist/kl_0.4/' 
    q_dataset = torch.load(dataset_path + 'train.pt')
    q_testset = torch.load(dataset_path + 'test.pt')
    result_path = root + '_mnist_results/' + direction + '/imb0.4/' + model_name
    path_classifier = root + '_mnist_results/_source_models/mnist_imb0.4'
    checkpoints_path_classifier = path_classifier + '/checkpoints/'
    checkpoints_path_adversarial = result_path + '/checkpoints/'

if imbalanced == False:
    result_path = root + '_mnist_results/' + direction + '/itzel/' + model_name
    path_classifier = root + '_mnist_results/_source_models/mnist'
    checkpoints_path_classifier = path_classifier + '/checkpoints_all/'
    checkpoints_path_adversarial = result_path + '/checkpoints/'
    
if not os.path.exists(checkpoints_path_adversarial):
    os.makedirs(checkpoints_path_adversarial)
    
if imbalanced == False:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
    ])

    MNIST_dataset = MNIST(root=root+'MNIST_scripts', train=True, download=False, transform=transform)
    MNIST_testset = MNIST(root=root+'MNIST_scripts', train=False, download=False, transform=transform)
    USPS_dataset = USPS(root=root+'USPS', train=True, download=False, transform=transform)
    USPS_testset = USPS(root=root+'USPS', train=False, download=False, transform=transform)

    if source == 'usps':
        split_frac = 0.95
        batch_size = 128
        train_size = round(len(USPS_dataset) * split_frac)
        val_size = len(USPS_dataset) - train_size
        q_dataset, q_valset = torch.utils.data.random_split(USPS_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        p_dataset = MNIST_dataset
        q_testset = USPS_testset
        p_testset = MNIST_testset
    if source == 'mnist':
    #     split_frac = 0.80
        batch_size = 1024
    #     train_size = round(len(MNIST_dataset) * split_frac)
    #     val_size = len(MNIST_dataset) - train_size
    #     q_dataset, q_valset = torch.utils.data.random_split(MNIST_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        q_dataset = MNIST_dataset
        p_dataset = USPS_dataset
        q_testset = MNIST_testset
        p_testset = USPS_testset

    # q_dataloader = {'train': DataLoader(q_dataset, batch_size=batch_size, shuffle=True),
    #            'val': DataLoader(q_valset, batch_size=batch_size, shuffle=True)} 
    
if imbalanced:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
    ])

    USPS_dataset = USPS(root=root+'USPS', train=True, download=False, transform=transform)
    USPS_testset = USPS(root=root+'USPS', train=False, download=False, transform=transform)
    p_dataset = USPS_dataset
    p_testset = USPS_testset
    batch_size = 1024
    
q_dataloader = DataLoader(q_dataset, batch_size=batch_size, shuffle=True)   
p_dataloader = DataLoader(p_dataset, batch_size=batch_size, shuffle=True)

q_testloader = DataLoader(q_testset, batch_size=batch_size, shuffle=True)
p_testloader = DataLoader(p_testset, batch_size=batch_size, shuffle=True)

p_iterations_per_epoch = len(p_dataset) / batch_size
q_iterations_per_epoch = len(q_dataset) / batch_size
iterations_per_epoch = math.ceil(min(p_iterations_per_epoch, q_iterations_per_epoch))

checkpoint_nr_lenet = 125
checkpoint_path_lenet = checkpoints_path_classifier + '/chkpt_epoch{}.pt'.format(checkpoint_nr_lenet)
checkpoint_lenet = torch.load(checkpoint_path_lenet, map_location=device)

if style == 'adda':
    p_lenet = networks_adda.LeNet(n_classes=10).to(device)
    p_lenet.load_state_dict(checkpoint_lenet['lenet_state_dict'])
    for name, param in p_lenet.classifier.named_parameters():
        param.requires_grad = False

    q_lenet = networks_adda.LeNet(n_classes=10).to(device)
    q_lenet.load_state_dict(checkpoint_lenet['lenet_state_dict'])
    for param in q_lenet.parameters():
        param.requires_grad = False
        
if style == 'shen':
    p_lenet = networks_adda.LeNet(n_classes=10).to(device)
    p_lenet.load_state_dict(checkpoint_lenet['lenet_state_dict'])
    
discriminator = networks_adda.Discriminator().to(device)
if weighted:
    weight = networks_adda.Weight(normalized=False, relu=False).to(device)
    
nu = 50
bp = 0.5
critic_repeats = 3
feature_repeats = 1
lr_disc = 1e-4 
lr_pfeat = 1e-4 
b1_disc = 0.5
b2_disc = 0.999
save_nr = 5
best_save = 50
checkpoint_start = 10
n_epochs = 3000
if weighted:
    lr_weight = 1e-4
# bce_loss = nn.BCELoss()
# loss = nn.CrossEntropyLoss()

if style == 'adda':
    opt_pfeat = torch.optim.Adam(p_lenet.feature_extractor.parameters(), lr=lr_pfeat)
if style == 'shen':
    opt_pfeat = torch.optim.Adam(p_lenet.parameters(), lr=lr_pfeat)
    
opt_disc = torch.optim.Adam(discriminator.parameters(), betas=(b1_disc, b2_disc), lr=lr_disc)
if weighted:
    opt_weight = torch.optim.Adam(weight.parameters(), betas=(b1_disc, b2_disc), lr=lr_weight)
    
def weights_init(x):
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        torch.nn.init.normal_(x.weight, 0.0, 0.001)
        torch.nn.init.constant_(x.bias, 0.1)
        
settings_dict = {'direction': direction,
                'len(p_dataset)': len(p_dataset),
                'len(q_dataset)': len(q_dataset),
                'discriminator':  discriminator,
                 'style': style,
                 'weight': weight,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'lr_disc': lr_disc,
                'lr_pfeat': lr_pfeat,
                'lr_weight': lr_weight,
                'opt_pfeat': opt_pfeat,
                'lr_disc': lr_disc,
                 'loss': 'wasserstein',
                 'nu': nu,
                 'critic_repeats': critic_repeats,
                 'feature_repeats': feature_repeats,
                 'checkpoint_nr_lenet': checkpoint_nr_lenet,
                 'bp': bp
               }

if not os.path.exists(result_path + '/adversarial_settings.pt'):
    torch.save(settings_dict, result_path + '/adversarial_settings.pt')
    
discriminator.apply(weights_init)
if weighted:
    weight.apply(weights_init)

disc_losses = []
weight_losses = []
pfeat_losses = []
q_classifier_accs = []
p_classifier_accs = []
q_classifier_losses = []

p_classifier_accs_test = []
q_classifier_accs_test = []

grad_penalties = []
grad_norms = []

best_model_test = copy.deepcopy(p_lenet.state_dict())
best_acc_test = (0,0)
best_model_train = copy.deepcopy(p_lenet.state_dict())
best_acc_train = (0,0)

since = time.time()

for epoch in range(n_epochs):
    
    q_correct_train = 0.0
    q_total_train = 0.0
    p_correct_train = 0.0
    p_total_train = 0.0
    
    for (p_data, p_classes), (q_data, q_classes) in zip(p_dataloader, q_dataloader):
        
        p_lenet.train()
        if weighted:
            weight.train()
        discriminator.train()
        
        # =============
        # TRAIN CRITIC
        # =============
        
        p_data = p_data.to(device)
        q_data = q_data.to(device)
        q_classes = q_classes.to(device)
        p_classes = p_classes.to(device)
        q_classes_onehot = F.one_hot(q_classes, num_classes=10).float()
        
        
        p_features = p_lenet.feature_extractor(p_data).to(device)
        if style == 'adda':
            q_features = q_lenet.feature_extractor(q_data).to(device)
        if style == 'shen':
            q_features = p_lenet.feature_extractor(q_data).to(device)
        if weighted:
            q_weights = weight(q_features)

        for i in range(critic_repeats):
            
            opt_disc.zero_grad()
            p_disc_pred = ( discriminator(p_features.detach()) )
            q_disc_pred = ( discriminator(q_features.detach()) )
            
            eps = torch.rand(size=[len(p_features),1],
                         device=device, requires_grad=True)
            differences = q_features[:len(p_features)] - p_features
            aux_features = p_features + (differences * eps)
            aux_features = aux_features.to(device)
            batch_avg_grad_norm = utils_adda.critic_gradient(aux_features, discriminator)
            grad_penalty = nu * ( (batch_avg_grad_norm-1)**2 ).mean()
            
            if weighted:
                disc_loss = p_disc_pred.mean() - (q_weights * q_disc_pred).mean() + grad_penalty
            else:
                disc_loss = p_disc_pred.mean() - q_disc_pred.mean() + grad_penalty    
            disc_loss.backward(retain_graph=True)
            opt_disc.step()
        disc_losses += [disc_loss.item()]
        grad_penalties += [grad_penalty.item()]
        grad_norms += [batch_avg_grad_norm.item()]

        
        # ==========================
        # TRAIN FEATURE & CLASSIFIER
        # ==========================

#         bce_loss = nn.BCELoss()

        for i in range(feature_repeats):
        
            p_features2 = p_lenet.feature_extractor(p_data).to(device)
            if style == 'adda':
                q_features2 = q_lenet.feature_extractor(q_data).to(device)
                q_logits = q_lenet.classifier(q_features2)
                
                q_features2_ = p_lenet.feature_extractor(q_data).to(device)
                q_logits_ = p_lenet.classifier(q_features2_)
            if style == 'shen':
                q_features2 = p_lenet.feature_extractor(q_data).to(device)
                q_logits = p_lenet.classifier(q_features2)
            
            if weighted:
                q_weights2 = weight(q_features2)
                bce_loss = nn.BCELoss(weight=q_weights2.detach())
            else:
                bce_loss = nn.BCELoss()
            q_sigmoid_preds = F.sigmoid(q_logits)
            q_classifier_loss = bce_loss(q_sigmoid_preds, q_classes_onehot)
 
            
            if style == 'adda':
                q_sigmoid_preds_ = F.sigmoid(q_logits_)
                q_classifier_loss_ = bce_loss(q_sigmoid_preds_, q_classes_onehot)

            p_disc_pred2 = ( discriminator(p_features2) )
            q_disc_pred2 = ( discriminator(q_features2) )
            
            if style == 'adda':
                wloss2 = - p_disc_pred2.mean() 
                pfeat_loss = wloss2 + q_classifier_loss_
            if style == 'shen':
                if weighted:
                    wloss2 = (q_weights2 * q_disc_pred2).mean() - p_disc_pred2.mean() 
                else:
                    wloss2 = q_disc_pred2.mean() - p_disc_pred2.mean()
                pfeat_loss = bp * wloss2 + q_classifier_loss
            opt_pfeat.zero_grad()
            pfeat_loss.backward()
            opt_pfeat.step()
        pfeat_losses += [pfeat_loss.item()]
        
#         _, q_preds = torch.max(q_logits,1)
#         q_correct_train += torch.sum(q_preds == q_classes.data).item()
#         q_total_train += len(q_classes)
#         q_classifier_losses += [q_classifier_loss.item()]

        # ==============
        # TRAIN WEIGHT
        # ==============
        if weighted:
            p_features3 = p_lenet.feature_extractor(p_data).to(device)
            q_features3 = p_lenet.feature_extractor(q_data).to(device)
            p_disc_pred3 = ( discriminator(p_features3) )
            q_disc_pred3 = ( discriminator(q_features3) )
            q_weights3 = weight(q_features3)

                     # classifier
            bce_loss3 = nn.BCELoss(weight=q_weights3.detach())
            q_logits = p_lenet.classifier(q_features3)
            q_sigmoid_preds = F.sigmoid(q_logits)
            q_classifier_loss = bce_loss3(q_sigmoid_preds, q_classes_onehot)

                  # feature
            wloss3 = (q_weights3 * q_disc_pred3).mean() - p_disc_pred3.mean()
            weight_loss =  wloss3 
            opt_weight.zero_grad()
            weight_loss.backward()
            opt_weight.step()
            weight_losses += [weight_loss.item()]
        
        
        max_values, q_preds = torch.max(q_logits,1)
        q_correct_train += torch.sum(q_preds == q_classes.data).item()
        q_total_train += len(q_classes)
        q_classifier_losses += [q_classifier_loss.item()]
        with torch.no_grad():

            p_logits = p_lenet(p_data)
            _, p_preds = torch.max(p_logits,1)
            p_correct_train += torch.sum(p_preds == p_classes.data).item()
            p_total_train += len(p_classes)
        

    # only compute acc over all data/iterations
    q_classifier_acc = q_correct_train / q_total_train
    q_classifier_accs.append(q_classifier_acc)
    p_classifier_acc = p_correct_train / p_total_train
    p_classifier_accs.append(p_classifier_acc)
    
    
    # =======================
    # TEST ACCURACY
    # =======================
    
    discriminator.eval()
    p_lenet.eval()
#     weight.eval()
        
    p_correct_test = 0.0
    p_total_test = 0.0
    q_correct_test = 0.0
    q_total_test = 0.0
    
    for (p_test, p_classes), (q_test, q_classes) in zip(p_testloader, q_testloader):
        
        p_data_test = p_test.to(device)
        q_data_test = q_test.to(device)
        
        p_classes_test = p_classes.to(device)
        q_classes_test = q_classes.to(device)
        
        with torch.no_grad():

            p_logits_test = p_lenet(p_data_test)
            if style == 'shen':
                q_logits_test = p_lenet(q_data_test)
            if style == 'adda':
                q_logits_test = q_lenet(q_data_test)
            
            _, p_preds_test = torch.max(p_logits_test,1)
            _, q_preds_test = torch.max(q_logits_test,1)
            
            p_correct_test += torch.sum(p_preds_test == p_classes_test.data).item()
            p_total_test += len(p_classes_test)
            q_correct_test += torch.sum(q_preds_test == q_classes_test.data).item()
            q_total_test += len(q_classes_test)

    # print the numbers after each epoch
    p_classifier_acc_test = p_correct_test / p_total_test
    p_classifier_accs_test.append(p_classifier_acc_test)
    
    q_classifier_acc_test = q_correct_test / q_total_test
    q_classifier_accs_test.append(q_classifier_acc_test)
    
    # check every epoch whether it's the best model
    if p_classifier_acc_test > best_acc_test[1]:
        best_acc_test = (epoch, p_classifier_acc_test)
        best_model_test = copy.deepcopy(p_lenet.state_dict())
    if p_classifier_acc > best_acc_train[1]:
        best_acc_train = (epoch, p_classifier_acc)
        best_model_train = copy.deepcopy(p_lenet.state_dict())
    
    
    if (epoch+1) > checkpoint_start and (epoch+1) % save_nr == 0:
        torch.save({
            'disc_state_dict': discriminator.state_dict(),
            'p_lenet_state_dict': p_lenet.state_dict(),
#             'weight_state_dict': weight.state_dict(),
            'pfeat_losses': pfeat_losses,
            'disc_losses': disc_losses,
#             'weight_losses': weight_losses,
            'grad_penalties': grad_penalties,
            'grad_norms': grad_norms,
            'q_classifier_accs': q_classifier_accs,
            'p_classifier_accs': p_classifier_accs,
            'q_classifier_losses': q_classifier_losses,
            'q_classifier_accs_test': q_classifier_accs_test,
            'p_classifier_accs_test': p_classifier_accs_test,

        }, checkpoints_path_adversarial + 'chkpt_epoch%d.pt' % (epoch + 1))
    
    # save best acc every 100 epochs
    if (epoch+1) % best_save == 0:
        torch.save({
            'best_acc_test': best_acc_test,
            'best_model_test': best_model_test,
            'best_acc_train': best_acc_train,
            'best_model_train': best_model_train
        }, checkpoints_path_adversarial + '_best_model%d.pt' % (epoch + 1))

    time_elapsed = time.time() - since
    print('Epoch: {}/{}'.format(epoch, n_epochs - 1))
    print('Already training for {:.0f}m {:.0f}s'.format(time_elapsed // 60,time_elapsed % 60))
    print(' ')
    print('Target feature loss: %f' % (pfeat_losses[-1]))
    print('Discriminator loss: %f' % (disc_losses[-1]))
#     print('Weight loss: %f' % (weight_losses[-1]))
    print('Source classifier loss: %f' % (q_classifier_losses[-1]))
    print('Source classifier acc: %f' % (q_classifier_accs[-1]))
    print('Target classifier acc: %f' % (p_classifier_accs[-1]))
    print(' ')
    print('Source test acc: %f' % (q_classifier_acc_test))
    print('Target test acc: %f' % (p_classifier_acc_test))
    print(' ')
    print('------------------------------')
    print(' ')

    
