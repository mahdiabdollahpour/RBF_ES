from deap import base, creator

import random
from deap import tools
# from fitness import evaluate
from params import *


class ES():
    def __init__(self, pop_size, evaluate, loss, x_data, y_data, IND_SIZE):
        # self.IND_SIZE = IND_SIZE
        self.pop_size = pop_size
        self.evaluate = evaluate
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        rand = lambda: np.random.normal(0, 1, 1)[0]
        self.toolbox.register("attribute", rand)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=10)
        self.toolbox.register("evaluate", self.evaluate, loss, x_data, y_data)
        self.eval = lambda x: self.evaluate(loss, x_data, y_data, x)

    def solve_problem(self, CXPB=0.2, MUTPB=0.3, NGEN=40):
        pop = self.toolbox.population(n=self.pop_size)
        # CXPB, MUTPB, NGEN = 0.5, 0.5, 40

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("--------------------------------------------")
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, self.pop_size)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop

    def get_best_in_pop(self, pop):
        return min(pop, key=self.eval)
