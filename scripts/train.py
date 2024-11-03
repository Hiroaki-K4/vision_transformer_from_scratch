import torch
import torch.nn as nn
import torchvision
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    # Load data
    #
    # We will use a subset of CIFAR10 dataset
    image_size = 224
    # define transform
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    torchvision.datasets.CIFAR10.url = (
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    )
    # init CIFAR10 training and test datasets
    trainset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    # get class names
    classes = trainset.classes
    # get a subset of the trainset and test set
    trainset = torch.utils.data.Subset(trainset, list(range(5000)))
    testset = torch.utils.data.Subset(testset, list(range(1000)))
    # output classes
    print(classes)

    # define data loaders
    batch_size = 16
    # percentage of training set to use as validation
    valid_size = 0.2
    # get training indices that wil be used for validation
    train_size = len(trainset)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * train_size))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers to obtain training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # prepare data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size)

    # print out classes statistics
    # get all training samples labels
    train_labels = [labels for i, (images, labels) in enumerate(train_loader)]
    train_labels = torch.cat((train_labels), 0)
    train_labels_count = train_labels.unique(return_counts=True)
    # print(train_labels_count)
    print("The number of samples per classes in training dataset:\n")
    for label, count in zip(train_labels_count[0], train_labels_count[1]):
        print("\t {}: {}".format(label, count))
    # get all test samples labels
    test_labels = [labels for i, (images, labels) in enumerate(test_loader)]
    test_labels = torch.cat((test_labels), 0)
    test_labels_count = test_labels.unique(return_counts=True)
    print()
    print("The number of samples per classes in test dataset:\n")
    for label, count in zip(test_labels_count[0], test_labels_count[1]):
        print("\t {}: {}".format(label, count))

    vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    print(vision_transformer)
    print(vision_transformer.heads)

    # fine-tune with dataset
    # change the number of output classes
    vision_transformer.heads = nn.Linear(
        in_features=768, out_features=len(classes), bias=True
    )
    # freeze the parameters except the last linear layer
    #
    # freeze weights
    for p in vision_transformer.parameters():
        p.requires_grad = False
    # unfreeze weights of classification head to train
    for p in vision_transformer.heads.parameters():
        p.requires_grad = True

    # check whether corresponding layers are frozen
    for layer_name, p in vision_transformer.named_parameters():
        print("Layer Name: {}, Frozen: {}".format(layer_name, not p.requires_grad))
        print()

    # specify loss function
    criterion = nn.CrossEntropyLoss()
    # define optimizer
    # only train the parameters with requires_grad set to True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, vision_transformer.parameters()), lr=0.0001
    )
    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    print("GPU available: ", train_on_gpu)

    # load model if it exists
    model_path = "vit.pth"
    if os.path.exists(model_path):
        vision_transformer.load_state_dict(torch.load(model_path))
    # Train model
    # number of epochs
    n_epoch = 5
    # number of iterations to save model
    n_step = 100
    train_loss_list, valid_loss_list = [], []
    # move model to GPU
    if train_on_gpu:
        vision_transformer.to("cuda")

    # prepare model for training
    vision_transformer.train()

    for e in tqdm(range(n_epoch)):
        train_loss = 0.0
        valid_loss = 0.0
        # get batch data
        for i, (images, targets) in enumerate(train_loader):
            # move to gpu if available
            if train_on_gpu:
                images, targets = images.to("cuda"), targets.to("cuda")
            # clear grad
            optimizer.zero_grad()
            # feedforward data
            outputs = vision_transformer(images)
            # calculate loss
            loss = criterion(outputs, targets)
            # backward pass, calculate gradients
            loss.backward()
            # update weights
            optimizer.step()
            # track loss
            train_loss += loss.item()
            # save the model parameters
            if i % n_step == 0:
                torch.save(vision_transformer.state_dict(), model_path)

        # set model to evaluation mode
        vision_transformer.eval()
        # validate model
        for images, targets in valid_loader:
            # move to gpu if available
            if train_on_gpu:
                images = images.to("cuda")
                targets = targets.to("cuda")
            # turn off gradients
            with torch.no_grad():
                outputs = vision_transformer(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        # set model back to trianing mode
        vision_transformer.train()
        # get average loss values
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        # output training statistics for epoch
        print(
            "Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}".format(
                (e + 1), train_loss, valid_loss
            )
        )

    # visualize loss statistics
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plot losses
    x = list(range(1, n_epoch + 1))
    plt.plot(x, train_loss_list, color="blue", label="Train")
    plt.plot(x, valid_loss_list, color="orange", label="Validation")
    plt.legend(loc="upper right")
    plt.xticks(x)
    plt.show()

    # prepare model for evaluation
    vision_transformer.eval()
    test_loss = 0.0
    accuracy = 0
    # number of classes
    n_class = len(classes)
    class_correct = np.zeros(n_class)
    class_total = np.zeros(n_class)
    # move model back to cpu
    vision_transformer = vision_transformer.to("cpu")
    # test model
    for images, targets in test_loader:
        # get outputs
        outputs = vision_transformer(images)
        # calculate loss
        loss = criterion(outputs, targets)
        # track loss
        test_loss += loss.item()
        # get predictions from probabilities
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        # get correct predictions
        correct_preds = (preds == targets).type(torch.FloatTensor)
        # calculate and accumulate accuracy
        accuracy += torch.mean(correct_preds).item() * 100
        # calculate test accuracy for each class
        for c in range(n_class):
            class_total[c] += (targets == c).sum()
            class_correct[c] += ((correct_preds) * (targets == c)).sum()

    # get average accuracy
    accuracy = accuracy / len(test_loader)
    # get average loss
    test_loss = test_loss / len(test_loader)
    # output test loss statistics
    print("Test Loss: {:.6f}".format(test_loss))

    class_accuracy = class_correct / class_total
    print("Test Accuracy of Classes")
    print()
    for c in range(n_class):
        print(
            "{}\t: {}% \t ({}/{})".format(
                classes[c],
                int(class_accuracy[c] * 100),
                int(class_correct[c]),
                int(class_total[c]),
            )
        )
    print()
    print(
        "Test Accuracy of Dataset: \t {}% \t ({}/{})".format(
            int(accuracy), int(np.sum(class_correct)), int(np.sum(class_total))
        )
    )


if __name__ == "__main__":
    main()
