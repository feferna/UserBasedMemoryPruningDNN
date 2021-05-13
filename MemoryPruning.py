import numpy as np
from individual import Individual
from copy import deepcopy


class MemoryPruning:
    def __init__(self, ideal_memory_footprint, flipping_prob, pop_size, mutation_p, train_parameters, model_parameters):
        self.ideal_memory_footprint = ideal_memory_footprint
        self.pop_size = pop_size
        self.flipping_prob = flipping_prob
        self.mutation_p = mutation_p

        self.train_parameters = train_parameters

        self.model_parameters = model_parameters

        self.individuals = [None] * self.pop_size

    def fit(self, run_number):
        # Initialize population
        for i in range(0, self.pop_size):
            self.individuals[i] = Individual(self.ideal_memory_footprint, self.flipping_prob, self.train_parameters,
                                             self.model_parameters)

            print("\tPreparing Individual: " + str(i))
            print("\t\tIndividual's IoU: " + str(self.individuals[i].fitness[0]))
            print("\t\tIndividual's Memory Footprint: " + str(self.individuals[i].fitness[1]))

        # Evaluate individuals
        IoU_arr = [x.fitness[0] for x in self.individuals]
        best_individual = deepcopy(self.individuals[np.argsort(IoU_arr)[-1]])

        best_individual.retrain(self.ideal_memory_footprint, run_number)

        print("")
        print("\tBest solution's memory footprint: " + str(best_individual.fitness[1]))
        print("\tMemory Decrease: " + str(best_individual.decrease_memory))
        print("\tBest solution's Mean IoU: " + str(best_individual.fitness[0]))
        print("")

        return best_individual
