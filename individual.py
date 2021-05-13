import torch
import torch.nn as nn

import numpy as np
import pickle

import pruning_utils

import os

import torch.autograd.profiler as profiler

from torchvision import models

from sklearn.metrics import jaccard_score as IoU


class Individual:
    def __init__(self, ideal_memory_footprint, flipping_prob, train_parameters, model_parameters):
        self.ideal_memory_footprint = ideal_memory_footprint
        self.model_parameters = model_parameters

        self.flipping_prob = flipping_prob
        self.backbone_name = model_parameters.backbone
        self.dataset_name = train_parameters.dataset_name
        self.train_loader = train_parameters.train_loader
        self.test_loader = train_parameters.test_loader

        self.partial_train_epochs = train_parameters.partial_train_epochs
        self.full_train_epochs = train_parameters.full_train_epochs
        self.learning_rate = train_parameters.learning_rate
        self.learning_rate_full_train = train_parameters.learning_rate_full_train

        self.knee = False
        self.min_acc = False
        self.min_time = False
        self.min_mem = False

        self.filters = create_layer_list(model_parameters.block_sizes, model_parameters.init_conv1_2_feat_maps,
                                         model_parameters.init_conv3_feat_maps)

        self.genes_sizes = [sum(self.filters[0].values()), sum(self.filters[1].values()), sum(self.filters[2].values())]

        self.genes = [np.ones(shape=(self.genes_sizes[0],), dtype=np.bool),
                      np.ones(shape=(self.genes_sizes[1],), dtype=np.bool),
                      np.ones(shape=(self.genes_sizes[2],), dtype=np.bool)]

        mutation0 = [model_parameters.init_mutation_prob] * len(self.genes[0])
        mutation1 = [model_parameters.init_mutation_prob] * len(self.genes[1])
        mutation2 = [model_parameters.init_mutation_prob] * len(self.genes[2])
        self.mutation = [mutation0, mutation1, mutation2]

        # Get the memory used by the original model
        original_model = self.load_initial_model()
        self.original_cpu_memory = self.compute_cpu_memory(original_model)

        self.get_to_ideal_memory()

        self.evaluate()

    def get_to_ideal_memory(self):
        self.evaluate_memory()

        while self.cpu_memory > self.ideal_memory_footprint:
            for i in range(len(self.genes[0])):
                if np.random.uniform() <= self.flipping_prob and self.genes[0][i] == 1:
                    self.genes[0][i] = not self.genes[0][i]

            for i in range(len(self.genes[1])):
                if np.random.uniform() <= self.flipping_prob and self.genes[1][i] == 1:
                    self.genes[1][i] = not self.genes[1][i]

            for i in range(len(self.genes[2])):
                if np.random.uniform() <= self.flipping_prob and self.genes[2][i] == 1:
                    self.genes[2][i] = not self.genes[2][i]

            self.evaluate_memory()

    def load_initial_model(self):
        if self.backbone_name == "ResNet50":
            model = models.segmentation.deeplabv3_resnet50(pretrained=True, aux_loss=True)
        else:
            model = models.segmentation.deeplabv3_resnet101(pretrained=True, aux_loss=True)

        if self.dataset_name == "SaoCarlos":
            model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

            checkpoint = torch.load(self.model_parameters.model_file)

            model.load_state_dict(checkpoint["model_state_dict"])

        elif self.dataset_name == "PennFudanPed":
            model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

            checkpoint = torch.load(self.model_parameters.model_file)

            model.load_state_dict(checkpoint["model_state_dict"])
        
        return model

    def compute_cpu_memory(self, model):
        model.to('cpu')
        model.eval()

        input_size = self.model_parameters.input_size
        test_input = torch.rand(input_size)

        # compute output
        with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            with profiler.record_function("model_inference"):
                with torch.no_grad():
                    model(test_input)

        # Memory is saved in Bytes
        cpu_memory = 0
        for i in range(len(prof.key_averages())):
            cpu_memory += prof.key_averages()[i].cpu_memory_usage

        return cpu_memory

    def decode_pruning(self, model):
        model = pruning_utils.residual_pruning(model, self.genes, self.filters)

        return model

    def evaluate_without_training(self):
        data_loader = self.train_loader

        # Copy original model
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        # Individual's training
        model.cuda()
        model.eval()

        criterion = nn.CrossEntropyLoss()

        running_loss = 0.0
        IoU_accumulated = [0.0, 0.0]
        num_samples = 0
        batch_counter = 0

        for i, (input, target) in enumerate(data_loader):
            with torch.no_grad():
                batch_counter += 1

                input_var = input.cuda()
                target_var = target.cuda()

                num_samples += len(target_var)

                # compute output
                model_outputs = model(input_var)['out']

                _, preds = torch.max(model_outputs, 1)

                batch_loss = criterion(model_outputs, target_var)

                running_loss += batch_loss.item()

                ground_truth = target_var.cpu().numpy().reshape(-1)
                predicted = preds.cpu().numpy().reshape(-1)

                IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)

        loss = running_loss / num_samples

        all_classes_IoU = IoU_accumulated / batch_counter
        mean_IoU = np.sum(all_classes_IoU[1:]) / (len(all_classes_IoU) - 1)

        self.cpu_memory = self.compute_cpu_memory(model)
        self.decrease_memory = ((self.original_cpu_memory - self.cpu_memory) / self.original_cpu_memory) * 100

        self.fitness = [mean_IoU, self.cpu_memory]

    def evaluate(self):
        data_loader = self.train_loader

        # Copy original model
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        # Individual's training
        model.train()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        model.cuda()

        loss_arr = []
        mean_IoU_arr = []
        mean_IoU = 1.0

        for epoch in range(self.partial_train_epochs):
            running_loss = 0.0
            IoU_accumulated = [0.0, 0.0]
            num_samples = 0
            batch_counter = 0

            for i, (input, target) in enumerate(data_loader):
                batch_counter += 1

                input_var = input.cuda()
                target_var = target.cuda()

                num_samples += len(target_var)

                # compute output
                optimizer.zero_grad()
                model_outputs = model(input_var)['out']

                _, preds = torch.max(model_outputs, 1)
                
                batch_loss = criterion(model_outputs, target_var)

                # Backward and optimize
                batch_loss.backward()
                optimizer.step()

                running_loss += batch_loss.item()

                ground_truth = target_var.cpu().numpy().reshape(-1)
                predicted = preds.cpu().numpy().reshape(-1)

                IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)

            epoch_loss = running_loss / num_samples
            loss_arr.append(epoch_loss)

            IoU_accumulated = IoU_accumulated / batch_counter
            mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)

            mean_IoU_arr.append(mean_IoU)

        self.cpu_memory = self.compute_cpu_memory(model)
        self.decrease_memory = ((self.original_cpu_memory - self.cpu_memory) / self.original_cpu_memory) * 100

        self.fitness = [mean_IoU, self.cpu_memory]
        print(self.fitness)

    def evaluate_memory(self):
        # Copy original model
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        self.cpu_memory = self.compute_cpu_memory(model)
        self.decrease_memory = ((self.original_cpu_memory - self.cpu_memory) / self.original_cpu_memory) * 100

        self.fitness = [0, self.cpu_memory]

    def retrain(self, ideal_memory_footprint, run_number):
        path_pruned_models = "./results/pruned_models/" + self.backbone_name + "/" + self.dataset_name + "/" +\
                             "MemoryFootprint_" + str(ideal_memory_footprint) + "/"

        if not os.path.exists(path_pruned_models):
            os.makedirs(path_pruned_models)

        data_loader = self.train_loader

        # Copy original model
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        # Individual's training
        model.train()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate_full_train,
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=7,
                                                       gamma=0.5)

        model.cuda()

        # ################
        # Train model
        # ################
        loss_arr = []
        mean_IoU_arr = []

        for epoch in range(self.full_train_epochs):
            print("Epoch: " + str(epoch))
            running_loss = 0
            IoU_accumulated = [0.0, 0.0]
            num_samples = 0
            batch_counter = 0
            for images, labels in data_loader:
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
            loss_arr.append(epoch_loss)

            IoU_accumulated = IoU_accumulated / batch_counter
            mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)

            mean_IoU_arr.append(mean_IoU)

            lr_scheduler.step()

            if epoch % 10 == 0:
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': epoch_loss,
                }, path_pruned_models + self.dataset_name + "_trained_pruned_" + self.backbone_name + "_" +
                   str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pth")

                # Save Model Representation
                with open(path_pruned_models + self.dataset_name + "_REPRESENTATION_" + self.backbone_name + "_" +
                          str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pickle", "wb") as f:
                    pickle.dump(self.genes, f)

                # Save Model Filters
                with open(path_pruned_models + self.dataset_name + "_FILTERS_" + self.backbone_name + "_" +
                          str(ideal_memory_footprint) + "_Run_Number_" + str(run_number) + ".pickle", "wb") as f:
                    pickle.dump(self.filters, f)


            print("\tMean Loss: {:.4f}".format(epoch_loss))
            print("\tMean IoU: {:.4f}".format(mean_IoU))

        # ############################
        # Test model in the test set
        # ############################
        IoU_accumulated = [0.0, 0.0]
        running_loss = 0
        batch_counter = 0
        num_samples = 0
        for images, labels in self.test_loader:
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

        print("\tTest Loss: {:.4f}".format(epoch_loss))
        print("\tTest IoU: {:.4f}".format(mean_IoU))

        self.cpu_memory = self.compute_cpu_memory(model)
        self.decrease_memory = ((self.original_cpu_memory - self.cpu_memory) / self.original_cpu_memory) * 100

        self.fitness = [mean_IoU, self.cpu_memory]


def create_layer_list(block_sizes, init_conv1_2_feat_maps, init_conv3_feat_maps):
    conv1_filters = {}
    conv2_filters = {}
    conv3_filters = {}

    conv1_filters['backbone.conv1'] = init_conv1_2_feat_maps

    conv1_2_feat_maps = init_conv1_2_feat_maps
    conv3_feat_maps = init_conv3_feat_maps

    for i in range(1, len(block_sizes) + 1):
        conv3_filters['backbone.layer' + str(i)] = conv3_feat_maps
        init_string = 'backbone.layer' + str(i) + "."

        for j in range(block_sizes[i-1]):
            conv1_string = init_string + str(j) + ".conv1"
            conv2_string = init_string + str(j) + ".conv2"

            conv1_filters[conv1_string] = conv1_2_feat_maps
            conv2_filters[conv2_string] = conv1_2_feat_maps
        
        conv1_2_feat_maps *= 2
        conv3_feat_maps *= 2

    return [conv1_filters, conv2_filters, conv3_filters]
