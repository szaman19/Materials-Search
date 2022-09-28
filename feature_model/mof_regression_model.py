from __future__ import print_function
from __future__ import division
import torch.nn as nn

import os
import pickle
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
from PIL import Image
import pandas as pd
import glob
import copy
from sklearn import preprocessing
import hiddenlayer as hl

np.random.seed(43)
torch.manual_seed(43)


def train_model2(model, trainloader, validloader, criterion, optimizer, scheduler, epochs, diff_lr=False,
                 device='cuda'):
    """
    Train the model and run inference on the validation dataset. Capture the loss
    of the trained model and validation model. Also display the accuracy of the
    validation model
    :param model - a pretrain model object
    :param trainloader - a generator object representing the train dataset
    :param validloader - a generator object representing the validation dataset
    :param criterion - a loss object
    :param optimizer - an optimizer object
    :param scheduler - a scheduler object that varies the learning rate every n epochs
    :param epochs - an integer specifying the number of epochs to train the model
    :param diff_lr - a boolean specifying whether to use differential learning rate
    :param device - a string specifying whether to use cuda or cpu
    return a trained model with the best weights
    """
    start = time.time()
    is_inception = True
    print_every = 50
    steps = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    valid_loss_min = np.Inf
    training_loss, validation_loss = [], []
    for epoch in range(epochs):
        lr_used = 0
        if diff_lr:
            for param in optimizer.param_groups:
                if param['lr'] > lr_used:
                    lr_used = param['lr']
            print('learning rate being used {}'.format(lr_used))
        running_loss = 0
        # train_acc = 0
        # \scheduler.step()
        model.train()
        for idx, (images, target) in enumerate(trainloader):
            steps += 1
            images, target = images.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and backward pass
            if is_inception:
                outputs, aux_outputs = model(images)
                outputs = np.squeeze(outputs)
                aux_outputs = np.squeeze(aux_outputs)
                loss1 = criterion(outputs, target)
                loss2 = criterion(aux_outputs, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(images)
                output = np.squeeze(output)
                loss = criterion(output, target)  # used to be target, output!!

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            # ps = torch.exp(output)
            # train_acc += (ps.max(dim=1)[1] == labels.data).type(torch.FloatTensor).mean()

        #            if steps % print_every == 0:
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            valid_loss = validation(model, validloader, criterion, device)

        # if test_accuracy > best_acc:
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch: {}/{}... ".format(epoch + 1, epochs),
              "Train MSE loss: {:.4f}".format(running_loss / len(trainloader.dataset)),
              "Validation MSE loss: {:.4f}".format(valid_loss / len(validloader.dataset)),
              )
        # save the losses
        training_loss.append(running_loss / len(trainloader.dataset))
        validation_loss.append(valid_loss / len(validloader.dataset))
        running_loss = 0

    print('Best validation MSE loss is {:.4f}'.format(valid_loss_min / len(validloader.dataset)))
    print('Time to complete training {} minutes'.format((time.time() - start) / 60))
    model.load_state_dict(best_model_wts)
    return model, training_loss, validation_loss


def validation(model, validloader, criterion=None, device='cuda'):
    """
    Compute loss on the validation dataset
    :param model - a pretrained model object
    :param validloader - a generator object representing the validataion dataset
    :param criterion - a loss object
    :param device - a string specifying whether to use cuda or cpu
    return a tuple of loss and accuracy
    """
    valid_loss = 0
    model.eval()
    for images, target in validloader:
        images, target = images.to(device), target.to(device)
        output = model(images)

        output = np.squeeze(output)
        target = np.squeeze(target)
        valid_loss += criterion(output, target).item() * images.size(0)  # used to be target, output

    return valid_loss


def parse_object_info(file):
    elements = pd.read_csv(os.path.join(image_dir, file), header=None)
    # print(elements)
    elements.columns = ['lcd']

    return elements


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class objectDataset(Dataset):
    def __init__(self, image_path, elements, transform):
        self.samples = []
        self.transform = transform
        self.elements = elements
        temp = ()

        for k, file in enumerate(glob.glob(image_path + "/*.png")):
            if file.endswith('.png'):

                image1 = Image.open(file)

                image2 = image1.convert('RGB')

                image22 = self.transform(image2)

                temp = (image22, self.elements['lcd'][k])

                self.samples.append(temp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def transform_images(size=299):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ]),
    }
    return data_transforms


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig('elements_array.png', dpi=300)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders):

    model.eval()

    out_labels = []
    out_list = []
    count = 0

    with torch.no_grad():
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # fig = plt.figure(figsize=(15, 8))

            outputs = model(inputs)

            for k in range(inputs.size()[0]):
                count += 1

                out_labels.append(labels[k].item())

                out_list.append(outputs[k][0].item())

        return out_list, out_labels




def optimizer(model, lr=0.001, weight_decay=1e-3 / 200):
    """
    Define the optimizer used to reduce the loss
    :param model - a pretrained model object
    :param lr - a floating point value defining the learning rate
    :param weight_decay - apply L2 regularization
    return an optimizer object
    """
    if model.__dict__['_modules'].get('fc', None):
        return optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    model_name = "inception"

    num_classes = 1

    batch_size = 64
    num_epochs = 0
    lr = 0.0001

    feature_extract = True

    image_dir = r'mof5_dataset'

    class_names2 = parse_object_info('mof5_info.txt')

    min_max_scaler = preprocessing.MinMaxScaler()

    norm = min_max_scaler.fit_transform(class_names2)

    objs = min_max_scaler.fit(class_names2)

    class_names = pd.DataFrame(norm)
    class_names.columns = ['lcd']

    image_transforms = transform_images(size=299)

    # dataset = objectDataset('./mof5_dataset', class_names, transform=image_transforms['test'])
    #
    # train_file = open('mof5_dataset.pt', "wb")
    # pickle.dump(dataset, train_file)
    #
    # train_file.close()


    # used for generating dataset on provided data
    
    dataset = objectDataset('./mof4_dataset', class_names, transform=image_transforms['test'])

    train_file = open('mof4_dataset.pt', "wb")
    pickle.dump(dataset, train_file)

    train_file.close()


    """
    # un-comment the line below to load the pickled dataset
    
    # dataset = pickle.load(open('mof_dataset1.pt', 'rb'))
    """

    # dataset = pickle.load(open('mof4_dataset.pt', 'rb'))

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    image_label, label = dataset[1]

    train_ds, val_ds = random_split(dataset, [8000, 4020], generator=torch.Generator().manual_seed(43))
    inputs, radius = [], []

    for i in range(len(train_ds)):
        inputs.append(train_ds[i][0])
        radius.append(train_ds[i][1])

    rads = pd.DataFrame(radius)

    u_rads = objs.inverse_transform(rads)

    urads = list(np.around(u_rads, 2))

    out = utils.make_grid(inputs[0:6])

    out_labels2 = []

    imshow(out, title=[urads[x][0] for x in range(6)])

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    criterion = nn.MSELoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    optim_ = optimizer(model_ft, lr)

    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=num_epochs, eta_min=0)

    # model_ft.load_state_dict(torch.load('final4_modelMOF.pt'))

    model_ft, training_loss, validation_loss = train_model2(model_ft, train_loader,
                                                            val_loader, criterion,
                                                            optim_, exp_lr_scheduler, num_epochs, False, device='cuda')

    # model_ft.load_state_dict(torch.load('final4_modelMOF.pt'))

    outlist = []

    out_list2, out_labels2 = visualize_model(model_ft, val_loader)
    #

    print(len(val_ds))
    print(len(out_list2))
    out_rel = []

    print("list: ", out_list2)
    print("labels: ", out_labels2)

    for z in range(len(val_ds)):
        out_rel.append(abs(out_list2[z] - out_labels2[z])/out_labels2[z])

    out_list3 = pd.DataFrame(out_list2)
    out_labels3 = pd.DataFrame(out_labels2)
    out_rel3 = pd.DataFrame(out_rel)


    fig, ax = plt.subplots(1)
    unnorm_list = objs.inverse_transform(out_list3)
    unnorm_labels = objs.inverse_transform(out_labels3)
    unnorm_rel = objs.inverse_transform(out_rel3)

    ax.hist(out_list2, color='gray', bins=100, range=(0, .4), alpha=0.5, label='Predicted MOF LCDs')
    ax.hist(out_labels2, color='green', bins=100, range=(0, .4), alpha=0.5, label='Actual MOF LCDs')

    ax.set_title("Predicted vs Actual MOF LCDs")
    ax.set(xlabel='Range of Standardized MOF LCDs', ylabel='Frequency of Predicted & Actual MOF LCDs')

    ax.legend(loc='upper right')
    ax.figure.savefig('graph4.png', dpi=300)

    # torch.save(model_ft.state_dict(), './final4_tsne_modelMOF.pt')

    fig2, ax3 = plt.subplots(1)
    ax3.hist(out_rel, color='orange', bins=100, alpha=0.5, label='Relative Error')
    ax3.set_title("Relative Error")
    ax3.set(xlabel='Range of Standardized Relative Error', ylabel='Frequency of Relative Error')

    ax3.legend(loc='upper right')
    ax3.figure.savefig('graph5.png', dpi=300)

    fig4, ax4 = plt.subplots(1)

    ax4.hist(unnorm_list, color='gray', bins=100, alpha=0.5, range=(0,40), label='Predicted MOF LCDs')
    ax4.hist(unnorm_labels, color='blue', bins=100, alpha=0.5, range=(0,40), label='Actual MOF LCDs')

    ax4.set_title("Predicted vs Actual MOF LCDs")
    ax4.set(xlabel='Range of True MOF LCDs', ylabel='Frequency of Predicted & Actual MOF LCDs')

    ax4.legend(loc='upper right')
    ax4.figure.savefig('graph6.png', dpi=300)

    average = sum(out_rel) / len(out_rel)
    merror = max(out_rel)

    print("relative error: ", out_rel[0:200])

    out_rel.sort()
    mid = len(out_rel) // 2
    res = (out_rel[mid] + out_rel[~mid]) / 2

    print("average: ", average)

    print("average %: ", average * 100, "%")

    print("max error: ", merror)

    print("max error %: ", merror * 100, "%")

    print("Median: ", str(res))
    print("Median %: ", str(res*100), "%")

    plt.show()
