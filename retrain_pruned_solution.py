import pickle
import torch
import torch.nn as nn

from torchvision import models

import numpy as np

import DataLoader
import pruning_utils

from sklearn.metrics import jaccard_score as IoU

import matplotlib.pyplot as plt


def load_initial_model(backbone_name, dataset_name):
    if backbone_name == "ResNet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, aux_loss=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, aux_loss=True)

    if dataset_name == "SaoCarlos":
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    elif dataset_name == "PennFudanPed":
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    return model


def decode_pruning(model, representation, filters):
    model = pruning_utils.residual_pruning(model, representation, filters)

    return model


def main():
    run_number = 28
    ideal_memory_footprint = 536870912
    backbone_name = "ResNet101"  # "ResNet50" or "ResNet101
    dataset_name = "SaoCarlos"  # or "PennFudanPed"
    batch_size = 32
    learning_rate = 0.000001
    number_epochs = 201

    train_loader, test_loader = DataLoader.data_loader(dataset_name, batch_size)

    path_pruned_model = "./results/pruned_models/" + backbone_name + "/" + dataset_name + "/" + "MemoryFootprint_" +\
                        str(ideal_memory_footprint) + "/" + dataset_name + "_trained_pruned_" + backbone_name + "_" +\
                        str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pth"

    path_pruned_model_representation = "./results/pruned_models/" + backbone_name + "/" + dataset_name + "/" +\
                                       "MemoryFootprint_" + str(ideal_memory_footprint) + "/" +\
                                       dataset_name + "_REPRESENTATION_" + backbone_name + "_" +\
                                       str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pickle"

    path_pruned_model_filters = "./results/pruned_models/" + backbone_name + "/" + dataset_name + "/" + \
                                "MemoryFootprint_" + str(ideal_memory_footprint) + "/" + \
                                dataset_name + "_FILTERS_" + backbone_name + "_" + \
                                str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pickle"

    with open(path_pruned_model_representation, 'rb') as f:
        model_representation = pickle.load(f)

    with open(path_pruned_model_filters, 'rb') as f:
        model_filters = pickle.load(f)

    # Copy original model
    model = load_initial_model(backbone_name, dataset_name)
    model = decode_pruning(model, model_representation, model_filters)

    checkpoint = torch.load(path_pruned_model)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Individual's training
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.5)

    #checkpoint = torch.load(path_pruned_model)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #lr_scheduler.load_state_dict(checkpoint['scheduler'])

    #e = checkpoint['epoch']
    #loss = checkpoint['loss']

    model.train()

    model.cuda()

    # ################
    # Train model
    # ################
    loss_arr = []
    train_IoU_arr = []
    test_IoU_arr = []
    # IoU_floodwall = []
    # IoU_river = []

    for epoch in range(0, number_epochs):
        print("Epoch: " + str(epoch))
        running_loss = 0
        # IoU_accumulated = [0.0, 0.0, 0.0]
        IoU_accumulated = [0.0, 0.0]
        num_samples = 0
        batch_counter = 0
        for images, labels in train_loader:
            batch_counter += 1
            images = images.cuda()
            labels = labels.cuda()

            num_samples += len(labels)

            model_outputs = model(images)['out']

            _, preds = torch.max(model_outputs, 1)

            batch_loss = criterion(model_outputs, labels)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            ground_truth = labels.cpu().numpy().reshape(-1)
            predicted = preds.cpu().numpy().reshape(-1)

            # IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1, 2], average=None)
            IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)

        epoch_loss = running_loss / num_samples
        loss_arr.append(epoch_loss)

        IoU_accumulated = IoU_accumulated / batch_counter
        mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)

        # IoU_floodwall.append(IoU_accumulated[1])
        # IoU_river.append(IoU_accumulated[2])
        train_IoU_arr.append(mean_IoU)

        lr_scheduler.step()

        if epoch % 10 == 0:
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': epoch_loss,
            }, "./results/retrained_pruned_model.pth")

        print("\tMean Loss: {:.4f}".format(epoch_loss))
        # print("\tMean IoU Floodwall: {:.4f}".format(IoU_accumulated[1]))
        # print("\tMean IoU River: {:.4f}".format(IoU_accumulated[2]))
        print("\tMean IoU: {:.4f}".format(mean_IoU))

        # ############################
        # Test model in the test set
        # ############################
        IoU_accumulated = [0.0, 0.0]
        running_loss = 0
        batch_counter = 0
        num_samples = 0
        for images, labels in test_loader:
            batch_counter += 1
            images = images.cuda()

            labels = labels.cuda()

            num_samples += len(labels)

            model_outputs = model(images)['out']

            _, preds = torch.max(model_outputs, 1)

            batch_loss = criterion(model_outputs, labels)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            ground_truth = labels.cpu().numpy().reshape(-1)
            predicted = preds.cpu().numpy().reshape(-1)

            IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)

        epoch_loss = running_loss / num_samples

        IoU_accumulated = IoU_accumulated / batch_counter
        mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)
        test_IoU_arr.append(mean_IoU)

        print("\tTest Loss: {:.4f}".format(epoch_loss))
        print("\tTest IoU: {:.4f}".format(mean_IoU))

    # Plot training graph
    fig, host = plt.subplots(nrows=1, ncols=1)

    par1 = host.twinx()

    host.set_xlabel("Epochs")
    host.set_ylabel("Loss")
    par1.set_ylabel("Intersection over Union (IoU)")

    p1, = host.plot(loss_arr, color='r', linestyle='--', marker='o', label="Training Loss")

    p2, = par1.plot(train_IoU_arr, color='b', linestyle='--', marker='*', label="Training IoU")

    p3, = par1.plot(test_IoU_arr, color='g', linestyle='--', marker='^', label="Test IoU")

    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='center right')
    #host.set_title("Retrained DeepLabv3 with " + backbone_name + " backbone pruned to 512MB on the Sao Carlos Dataset")

    plt.savefig(dataset_name + "_training_" + backbone_name + ".pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
