import random
import numpy as np
import array
import random
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def find_circle_coordinates(x_train, y_train, number_of_circles, evaluator, loss, NGEN=10):
    dim = len(x_train[0])
    chromosome_size = (dim + 1) * number_of_circles

    es = NewES(evaluator, loss, x_train, y_train, chromosome_size)

    pop = es.solve_problem(NGEN=NGEN)
    best = es.get_best_in_pop(pop)
    return best


class NewES():
    def __init__(self, evaluate, loss, x_data, y_data, IND_SIZE, MIN_VALUE=-1.5, MAX_VALUE=1.5, MIN_STRATEGY=0.5,
                 MAX_STRATEGY=3):
        self.evaluate = evaluate
        self.eval = lambda x: self.evaluate(loss, x_data, y_data, x_data, y_data, x)
        # self.toolbox.register("evaluate", self.eval)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
        creator.create("Strategy", array.array, typecode="d")
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generateES, creator.Individual, creator.Strategy,
                         IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxESBlend, alpha=0.1)
        self.toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.eval)

        self.toolbox.decorate("mate", self.checkStrategy(MIN_STRATEGY))
        self.toolbox.decorate("mutate", self.checkStrategy(MIN_STRATEGY))
    def generateES(self, icls, scls, size, imin, imax, smin, smax):
        ind = icls(random.uniform(imin, imax) for _ in range(size))
        ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
        return ind

    def checkStrategy(self,minstrategy):
        def decorator(func):
            def wrappper(*args, **kargs):
                children = func(*args, **kargs)
                for child in children:
                    for i, s in enumerate(child.strategy):
                        if s < minstrategy:
                            child.strategy[i] = minstrategy
                return children

            return wrappper

        return decorator

    def solve_problem(self, MU=10, LAMBDA=100, NGEN=30):
        MU, LAMBDA = 10, 100
        pop = self.toolbox.population(n=MU)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaMuCommaLambda(pop, self.toolbox, mu=MU, lambda_=LAMBDA,
                                                  cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=stats, halloffame=hof)

        return pop, logbook, hof

    def get_best_in_pop(self, pop):
        return min(pop, key=self.eval)
