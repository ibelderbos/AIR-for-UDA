import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
import warnings
warnings.filterwarnings('ignore')
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
torch.manual_seed(0)
import networks
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False
root = '/opt/app-root/s3_home/'

source = 'mnist'
target = 'usps'
model_name = 'exp009_unw'
style = 'shen'
direction = source + '_' + target
imbalanced = False
weighted = False

if imbalanced:
    dataset_path = root + '_subsets/mnist/kl_0.74/' 
    q_dataset = torch.load(dataset_path + 'train.pt')
    q_testset = torch.load(dataset_path + 'test.pt')
    result_path = root + '_mnist_results/' + direction + '/imb0.74/' + model_name
    path_classifier = root + '_mnist_results/_source_models/mnist_imb0.74'
    checkpoints_path_lenet = path_classifier + '/checkpoints/'
    checkpoints_path_adversarial = result_path + '/checkpoints/'
    
if imbalanced == False:
    result_path = root + '_mnist_results/' + direction + '/itzel/' + model_name
    path_classifier = root + '_mnist_results/_source_models/mnist'
    checkpoints_path_lenet = path_classifier + '/checkpoints_all/'
    checkpoints_path_adversarial = result_path + '/checkpoints/'
    
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
    
# settings_lenet = torch.load(path_classifier + 'cnn_settings_all.pt')
settings_adv = torch.load(result_path + '/adversarial_settings.pt')
batch_size = settings_adv['batch_size']
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28)),
])

q_dataloader = DataLoader(q_dataset, batch_size=batch_size, shuffle=True)   
p_dataloader = DataLoader(p_dataset, batch_size=batch_size, shuffle=True)

q_testloader = DataLoader(q_testset, batch_size=batch_size, shuffle=True) 
p_testloader = DataLoader(p_testset, batch_size=batch_size, shuffle=True)

p_iterations_per_epoch = len(p_dataset) / batch_size
q_iterations_per_epoch = len(q_dataset) / batch_size
iterations_per_epoch = math.ceil(min(p_iterations_per_epoch, q_iterations_per_epoch))

checkpoint_nrs_adv = [int(i.strip('.pt').strip('chkpt_epoch')) for i in os.listdir(checkpoints_path_adversarial) if not i.startswith('_')] 
print('Latest feature checkpoints: ',sorted(checkpoint_nrs_adv)[-5:])

checkpoint_nr_best = [int(i.strip('.pt').strip('_best_model')) for i in os.listdir(checkpoints_path_adversarial) if i.startswith('_')] 
print('Best feature checkpoints: ',sorted(checkpoint_nr_best))

checkpoint_best = torch.load(checkpoints_path_adversarial + '_best_model300.pt', map_location=device)

best_acc_test = checkpoint_best['best_acc_test']
best_model_test = checkpoint_best['best_model_test']
best_acc_train = checkpoint_best['best_acc_train']
best_model_train = checkpoint_best['best_model_train']

checkpoint_nr_adversarial = 300
checkpoint_path_adversarial1 = checkpoints_path_adversarial + '/chkpt_epoch{}.pt'.format(checkpoint_nr_adversarial)
checkpoint_adv = torch.load(checkpoint_path_adversarial1, map_location=device)
checkpoint_adv.keys()

pfeat_losses = checkpoint_adv['pfeat_losses'] 
disc_losses = checkpoint_adv['disc_losses'] 
grad_penalties = checkpoint_adv['grad_penalties']
grad_norms = checkpoint_adv['grad_norms']
q_classifier_accs = checkpoint_adv['q_classifier_accs']
p_classifier_accs = checkpoint_adv['p_classifier_accs']
q_classifier_losses = checkpoint_adv['q_classifier_losses']
p_classifier_accs_test = checkpoint_adv['p_classifier_accs_test']
q_classifier_accs_test = checkpoint_adv['q_classifier_accs_test']
if weighted:
    weight_losses = checkpoint_adv['weight_losses']

critic_repeats = settings_adv['critic_repeats']
epochs = list(range(0,2000,iterations_per_epoch))

plt.rcParams['figure.figsize'] = (10,5)
# plt.plot(p_classifier_accs_test, label='Target test accuracy')
# plt.plot(q_classifier_accs_test, label='Source test accuracy')
plt.plot(q_classifier_accs, label='Source accuracy')
plt.plot(p_classifier_accs[2:], label='Target accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend();

plt.plot(pfeat_losses, label='feature p losses', alpha=0.5)
plt.plot(disc_losses, label='critic losses')
plt.plot(q_classifier_losses, label='classifier losses', alpha=0.5)
# plt.plot(grad_norms, label ='grad norms')
plt.plot(grad_penalties, label ='grad penalties')
plt.plot([i-j for (i,j) in zip(disc_losses, grad_penalties)], label='critic loss without penalties')
if weighted:
    plt.plot(weight_losses, label = 'weight losses', alpha=0.4, linestyle='--')

# plt.xlim(0,80000)
# plt.ylim(-1,1)
plt.xlabel('Iterations')
plt.legend()
plt.title('Losses over iterations');

# checkpoint_nr_lenet = settings_adv['checkpoint_nr_lenet']
checkpoint_nr_lenet = 80
spec_checkpoints_path_lenet = checkpoints_path_lenet + 'chkpt_epoch{}.pt'.format(checkpoint_nr_lenet)
checkpoint_lenet = torch.load(spec_checkpoints_path_lenet, map_location=device)

# UNTIED WEIGHTS ==> SHEN
p_lenet = networks_adda.LeNet(n_classes=10)
q_lenet = networks_adda.LeNet(n_classes=10)

# load in the trained lenet params, particularly the classifier => just for comparison
q_lenet.load_state_dict(checkpoint_lenet['lenet_state_dict'])

# substitute the feature extractor for the target parameters
# p_lenet.load_state_dict(checkpoint_adv['p_lenet_state_dict'])
p_lenet.load_state_dict(checkpoint_best['best_model_test'])

# p_lenet.load_state_dict(checkpoint_lenet['lenet_state_dict'])
# p_lenet.feature_extractor.load_state_dict(checkpoint_adv['p_feature_state_dict'])

q_lenet = q_lenet.to(device)
q_lenet.eval()
p_lenet = p_lenet.to(device)
p_lenet.eval();

def test_prediction(dataset, dataloader, classif_model):
    all_classif = []
    all_classes = []
    all_embeddings = torch.empty([len(dataset), 500])
    correct = 0.0
    total = 0.0
    it_count = 0
    for inputs, classes in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            classes = classes.to(device)
                        
            outputs = classif_model(inputs)
            embed_feature = classif_model.feature_extractor(inputs)
            max_values, batch_preds = torch.max(outputs,1)

            correct += torch.sum(batch_preds == classes.data).item()
            total += len(classes)

        all_classif.append(batch_preds)
        all_classes.append(classes)
        all_embeddings[len(inputs)*it_count:len(inputs)*(it_count+1),:] = embed_feature
        it_count += 1
    epoch_acc = correct/total
    
    all_classif_flat = [int(item.cpu()) for sublist in all_classif for item in sublist]
    all_classes_flat = [int(item.cpu()) for sublist in all_classes for item in sublist]
    
    return epoch_acc, all_classif_flat, all_classes_flat, all_embeddings

p_epoch_acc, p_classif, p_classes, p_embeddings = test_prediction(p_testset, p_testloader, p_lenet)
p_epoch_acc_tr, p_classif_tr, p_classes_tr, p_embeddings_tr = test_prediction(p_dataset, p_dataloader, p_lenet)
p_epoch_acc_s, p_classif_s, p_classes_s, p_embeddings_s = test_prediction(p_testset, p_testloader, q_lenet)
q_epoch_acc, q_classif, q_classes, q_embeddings = test_prediction(q_testset, q_testloader, q_lenet)

print('Target acc test:           %.4f' % p_epoch_acc)
print('Target acc train:          %.4f' % p_epoch_acc_tr)
print('Target acc source only:    %.4f' % p_epoch_acc_s)
print('Source acc:                %.4f' % q_epoch_acc)