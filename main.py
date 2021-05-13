from DataLoader import data_loader
from MemoryPruning import *
import matplotlib.pyplot as plt


class TrainParameters:
    def __init__(self, dataset_name, train_loader, test_loader,
                 partial_train_epochs, full_train_epochs, learning_rate, learning_rate_full_train):
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.partial_train_epochs = partial_train_epochs
        self.full_train_epochs = full_train_epochs
        self.learning_rate = learning_rate
        self.learning_rate_full_train = learning_rate_full_train


class ModelParameters:
    def __init__(self, dataset_name, backbone, init_mutation_prob):
        self.backbone = backbone
        self.init_mutation_prob = init_mutation_prob
        self.input_size = (1, 3, 224, 224)

        if self.backbone == "ResNet50":
            self.block_sizes = [3, 4, 6, 3]  # Define the number of layers for each block of layers
        else:
            self.block_sizes = [3, 4, 23, 3]  # Define the number of layers for each block of layers

        self.model_file = "./original_models/" + dataset_name + "_trained_model_" + self.backbone + ".pth"

        self.init_conv1_2_feat_maps = 64
        self.init_conv3_feat_maps = 256


def run_alg(ideal_memory_footprint, flipping_prob, number_trials, pop_size, train_parameters, model_parameters):
    best_solutions = [None] * number_trials
    best_IoU_arr = [0.0] * number_trials
    best_memory_arr = [0.0] * number_trials

    for n in range(number_trials):
        print("Run number: " + str(n))
        memory_pruning_ga = MemoryPruning(ideal_memory_footprint, flipping_prob, pop_size, mutation_p,
                                            train_parameters, model_parameters)

        best_solutions[n] = memory_pruning_ga.fit(n)
        best_IoU_arr[n] = best_solutions[n].fitness[0]
        best_memory_arr[n] = best_solutions[n].fitness[1]

        print(best_IoU_arr[n])
        print(best_memory_arr[n])

    # Plot best IoU and memory over the course of generations
    fig1, ax1 = plt.subplots(nrows=1, ncols=2)

    fig1.suptitle("DeepLabv3 - " + model_parameters.backbone + " - " + train_parameters.dataset_name)

    ax1[0].boxplot([best_IoU_arr])
    ax1[0].set_xticks([1])
    ax1[0].set_xticklabels(["Mean IoU"])

    ax1[1].boxplot([best_memory_arr])
    ax1[1].set_xticks([1])
    ax1[1].set_xticklabels(["Memory Footprint in Bytes"])

    fig1.tight_layout()

    plt.savefig("./results/Boxplots - " + train_parameters.dataset_name + " - " + model_parameters.backbone + ".pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    # IDEAL_MEMORY_FOOTPRINT = 1073741824  # 1GB
    IDEAL_MEMORY_FOOTPRINT = 536870912  # 512MB

    DATASET = "SaoCarlos"
    # DATASET = "PennFudanPed"

    # BACKBONE = "ResNet50"
    BACKBONE = "ResNet101"

    POP_SIZE = 30

    NUMBER_TRIALS = 30

    MUTATION_P = 0.2  # Probability of mutation
    FLIPPING_P = 0.1

    PARTIAL_TRAIN_EPOCHS = 1
    FULL_TRAIN_EPOCHS = 101
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    LEARNING_RATE_FULL_TRAIN = 0.00001

    TRAIN_LOADER, TEST_LOADER = data_loader(DATASET, BATCH_SIZE)

    TRAIN_PARAM = TrainParameters(DATASET, TRAIN_LOADER, TEST_LOADER, PARTIAL_TRAIN_EPOCHS, FULL_TRAIN_EPOCHS,
                                  LEARNING_RATE, LEARNING_RATE_FULL_TRAIN)

    MODEL_PARAM = ModelParameters(DATASET, BACKBONE, MUTATION_P)

    run_alg(IDEAL_MEMORY_FOOTPRINT, FLIPPING_P, NUMBER_TRIALS, POP_SIZE, TRAIN_PARAM, MODEL_PARAM)
