import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import random

def get_office_loader(dataset_path, batch_size, split_val=False, split_frac=0.8, shuffle=True):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    dataset = datasets.ImageFolder(dataset_path, transform)

       
    if split_val:
        train_size = round(len(dataset) * split_frac)
        val_size = len(dataset) - train_size
        train, val = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        dataloader = {'train': DataLoader(train, batch_size=batch_size, shuffle=shuffle),
               'val': DataLoader(val, batch_size=batch_size, shuffle=shuffle)}
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, dataset

class CustomFeatureDataset(Dataset):
    def __init__(self, labels, data, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.data[index]
        y_label = self.labels[index]
        
        return (image, y_label)

def critic_gradient(aux_data, critic):

    aux_pred = critic(aux_data)
    grad = torch.autograd.grad(
        inputs = aux_data,
        outputs = aux_pred,
        grad_outputs = torch.ones_like(aux_pred),
        create_graph = True,
        retain_graph = True
    )[0]
    
    grad_flat = grad.view(len(grad),-1)
    grad_norms = (grad_flat.norm(2,dim=1))
    batch_avg_grad_norm = grad_norms.mean()
    
    return batch_avg_grad_norm

def show_images(nr_images, dataset, n_channels):

    size1 = dataset[0][0].shape[1]
    size2 = dataset[0][0].shape[2]
    imgs = torch.empty(nr_images, n_channels, size1, size2)
    for i in range(nr_images):
        img_idx = random.randrange(0,len(dataset))
        image = dataset[img_idx][0]
        imgs[i] = image
        
    nrow = int(np.sqrt(nr_images))
    grid_img = make_grid(tensor=imgs.data,
                         nrow=nrow,
                         normalize=True)
    grid_img = grid_img.permute(1,2,0)
    plt.axis('off')
    plt.rcParams['figure.figsize'] = (5,5)
    plt.imshow(grid_img)
    
def show_image(img, title):
    # transform from (-1,1) to (0,1)
    img = (img + 1) / 2
    img = img.detach().cpu()
    # plt.imshow(img.permute(1,2,0).squeeze())
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show

# based on shuffled dataloader    
def get_all_labels(dataloader, dataset, batch_size, device):
    all_labels = torch.empty([len(dataset)])
    for i, batch in enumerate(dataloader):
        labels = batch[1].to(device)
        size_cur_batch = len(labels)
        all_labels[i*batch_size:i*batch_size+size_cur_batch] = labels
    return all_labels

def get_freq_labels(dataset, n_classes):
    freq = torch.empty((n_classes))
    for i in dataset.targets:
        freq[int(i)] = freq[int(i)] + 1
    return freq

class CustomDataset(Dataset):
    def __init__(self, labels, data, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.data[index][0]
#         image = image.repeat(3,1,1) # convert to RGB
        y_label = self.labels[index]
        
        return (image, y_label)
    
class CustomSubset(Dataset):
    
    def __init__(self, all_dataset, data_indices_subset, labels):
        self.dataset = all_dataset
        self.indices = data_indices_subset
        self.labels = labels
        
    def __getitem__(self, subset_idx):
        image = self.dataset[self.indices[subset_idx]][0]
        label = self.labels[self.indices[subset_idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor