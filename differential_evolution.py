import numpy as np
import copy
import math
from random import uniform, sample, randint, random
from random import seed as seeding
import math
import time
from scipy.special import expit
from itertools import product


class OptimizationResults(dict):
    """
    Vessel for optimization results
    """

    def __init__(self,
                 result=None,
                 x=None,
                 fun=None,
                 niter=None,
                 final_fitness=None,
                 final_generation=None):

        self['result'] = result
        self['x'] = x
        self['fun'] = fun
        self['niter'] = niter
        self['final_fitness'] = final_fitness
        self['final_generation'] = final_generation

    @property
    def result(self):
        return self['result']

    @property
    def x(self):
        return self['x']

    @property
    def fun(self):
        return self['fun']

    @property
    def niter(self):
        return self['niter']

    @property
    def final_fitness(self):
        return self['final_fitness']

    @property
    def final_generation(self):
        return self['final_generation']


class DifferentialEvolution:
    """
    Class for DE solving
    """

    def __init__(self):

        self._initial_generation = None
        self._current_generation = None
        self._bounds = None
        self._current_fitness = None
        self._func = None

        self._std_fitness = None
        self._mean_fitness = None
        self._min_fitness = None
        self._max_fitness = None

        self._population = None
        self._ratios = None

    def set_bounds(self, bounds):

        self._bounds = bounds

    def generate_initial_population(self, size=30, distribution=uniform, seed=None):

        # This is random.seed with different name
        seeding(seed)

        # Initialize
        initial_generation = [None] * size

        # Looping
        for i in range(size):

            new_member = np.array([distribution(bound[0], bound[1]) for bound in self._bounds])
            initial_generation[i] = new_member

        self._initial_generation = initial_generation

        return initial_generation

    def best_evolve_generation(self, p=0.05, mutation_limits=None, crossover_limits=None):

        updates = 0

        # Best p:th percentile
        length = len(self._current_generation)
        p_stop = int(math.ceil(length * p))
        p_limit = sorted(self._current_fitness)[p_stop]

        generation = copy.copy(self._current_generation)
        fitness = copy.copy(self._current_fitness)

        for i, member in enumerate(generation):

            # Randomize mutation and crossover constants
            mutation = uniform(mutation_limits[0], mutation_limits[1])
            crossover = uniform(crossover_limits[0], crossover_limits[1])

            # Pick a member from p:th percentile which is different from the one to evolve
            p_idx = sample([idx for idx, fitness in enumerate(self._current_fitness) if fitness <= p_limit and not idx == i], 1)
            best_p = self._current_generation[p_idx[0]]

            # Pick random vectors
            a, b, c = sample([one for j, one in enumerate(self._current_generation) if j not in (i, p_idx)], 3)
            new_try = a + mutation * (best_p - a) + mutation * (b - c)

            # Random position for the vector
            r = randint(0, len(member) - 1)

            for k in range(len(member)):

                if k != r and random() > crossover:
                    new_try[k] = member[k]

                else:
                    min_bound, max_bound = self._bounds[k]

                    if new_try[k] < min_bound:
                        new_try[k] = 2 * min_bound - new_try[k]

                    elif new_try[k] > max_bound:
                        new_try[k] = 2 * max_bound - new_try[k]

            new_fitness = self._func(new_try)

            if new_fitness < self._current_fitness[i]:

                generation[i] = new_try
                fitness[i] = new_fitness
                updates += 1

        self._current_generation = generation
        self._current_fitness = fitness

        return updates, length - updates

    def random_evolve_generation(self, mutation_limits=None, crossover_limits=None):

        length = len(self._current_generation)
        updates = 0

        generation = copy.copy(self._current_generation)
        fitness = copy.copy(self._current_fitness)

        for i, member in enumerate(generation):

            # Randomize mutation and crossover constants
            mutation = uniform(mutation_limits[0], mutation_limits[1])
            crossover = uniform(crossover_limits[0], crossover_limits[1])

            a, b, c, d, e = sample([one for j, one in enumerate(self._current_generation) if not i == j], 5)
            new_try = a + mutation * (b - c) + mutation * (d - e)

            r = randint(0, len(member) - 1)

            for k in range(len(member)):

                if k != r and random() > crossover:
                    new_try[k] = member[k]

                else:
                    min_bound, max_bound = self._bounds[k]

                    new_try[k] = min(max_bound, max(min_bound, new_try[k]))

            new_fitness = self._func(new_try)

            # Update
            if new_fitness < self._current_fitness[i]:

                generation[i] = new_try
                fitness[i] = new_fitness
                updates += 1

        self._current_generation = generation
        self._current_fitness = fitness

        return updates, length - updates

    def optimize(self,
                 func,
                 tol=0.01,
                 generations=1000,
                 min_population=32,
                 culling=(0.2, 0.6),
                 generational_cycle=10,
                 seed=None):

        # This is random.seed with different name
        seeding(seed)

        self._func = func

        self._current_generation = copy.deepcopy(self._initial_generation)
        self._current_fitness = [self._func(x) for x in self._current_generation]

        # Initialize information vectors
        self._std_fitness = np.zeros(generations + 1)
        self._mean_fitness = np.zeros(generations + 1)
        self._min_fitness = np.zeros(generations + 1)
        self._max_fitness = np.zeros(generations + 1)

        # Set initial values
        self._std_fitness[0] = np.std(self._current_fitness)
        self._mean_fitness[0] = np.mean(self._current_fitness)
        self._min_fitness[0] = min(self._current_fitness)
        self._max_fitness[0] = max(self._current_fitness)

        self._population = [len(self._current_generation)]
        self._ratios = []
        ratio = 0

        start = time.time()

        strategy = 'exploration'

        # The loop
        for n_round in range(generations):

            total_updates, total_failures = 0, 0

            # Choose strategy
            if strategy != 'exploration':

                # For convergence
                mutation_limits = (0.3, 0.6)
                crossover_limits = (0.5, 0.7)

            else:

                # For exploration
                mutation_limits = (0.8, 1.5)
                crossover_limits = (0.05, 0.2)

            # Choose new vector generation strategy
            if random() < n_round / generations:
                updates, failures = self.best_evolve_generation(p=0.1,
                                                                mutation_limits=mutation_limits,
                                                                crossover_limits=crossover_limits)

            else:
                updates, failures = self.random_evolve_generation(mutation_limits=mutation_limits,
                                                                  crossover_limits=crossover_limits)

            # Update information

            total_updates += updates
            total_failures += failures

            current_std = np.std(self._current_fitness)
            current_mean = np.mean(self._current_fitness)

            self._std_fitness[n_round + 1] = current_std
            self._mean_fitness[n_round + 1] = current_mean
            self._min_fitness[n_round + 1] = min(self._current_fitness)
            self._max_fitness[n_round + 1] = max(self._current_fitness)

            # Check for stopping
            if current_mean == 0 or abs(current_std / current_mean) < tol:
                print('Finished!')

                n_iter = n_round

                self._std_fitness = self._std_fitness[:n_round + 2]
                self._mean_fitness = self._mean_fitness[:n_round + 2]
                self._min_fitness = self._min_fitness[:n_round + 2]
                self._max_fitness = self._max_fitness[:n_round + 2]

                minimi = min(self._current_fitness)

                xs = [self._current_generation[idx] for idx, fitness in enumerate(self._current_fitness) if fitness == minimi]

                return OptimizationResults(x=xs,
                                           result=True,
                                           fun=minimi,
                                           niter=n_iter,
                                           final_generation=self._current_generation,
                                           final_fitness=self._current_fitness)

            # For every cycle, we do maintenance work
            elif not n_round % generational_cycle:

                ratio = total_updates / (total_updates + total_failures)
                self._ratios.append(ratio)

                min_fitness = round(min(self._current_fitness), 4)
                mean_fitness = round(current_mean, 5)
                diversity = round(abs(current_std / current_mean), 5)
                size = len(self._current_generation)
                end = time.time()
                duration = np.round(end - start, 2)
                ratio = round(ratio, 3)
                print_line = 'Generation: {:=5}. Size {:=4}. Best fitness: {:8.4f}. Mean fitness: {:8.4f}. Diversity: {:8.4f}. Update ratio: {:5.4f}. Strategy: {}. Seconds: {:5}'
                print(print_line.format(n_round,
                                        size,
                                        min_fitness,
                                        mean_fitness,
                                        diversity,
                                        ratio,
                                        strategy,
                                        duration))

                if strategy == 'exploration':
                    # Toward the end, we prefer convergence, or if the exploration does not lead to gains
                    if random() > min(ratio, 1 - n_round / generations):
                        strategy = 'convergence'

                else:
                    # Earlier, we prefer exploration
                    if random() > max(ratio, n_round / generations):
                        strategy = 'exploration'

                # If there were very few updates, we ween the population
                if n_round > 0 and ratio < max(0.1, n_round / generations) and self._population[-1] > min_population:

                    took = (culling[0] + (1 - ratio) * (culling[1] - culling[0])) * len(self._current_fitness)

                    take = max(1, math.ceil(took))

                    n = min(take, len(self._current_fitness) - min_population)

                    if n > 0:

                        # Find the worst
                        idx = np.argsort(self._current_fitness)[-n:]

                        self._current_generation = [x for i, x in enumerate(self._current_generation) if i not in idx]
                        self._current_fitness = [x for i, x in enumerate(self._current_fitness) if i not in idx]

                start = time.time()

        else:

            # Stopping algorithm
            n_iter = generations
            minimi = min(self._current_fitness)

            xs = [self._current_generation[idx] for idx, fitness in enumerate(self._current_fitness) if fitness == minimi]

            return OptimizationResults(x=xs,
                                       result=False,
                                       fun=minimi,
                                       niter=n_iter)


"""
solver = DifferentialEvolution()
solver.set_bounds([(-5, 5)]*50)
solver.generate_initial_population(size=500)


def fun_fact(n):

    rand_vec = [2 * random() - 1 for i in range(n)]

    def fun(x):

        return sum([(x[i] - rand_vec[i])**2 for i, _ in enumerate(x)])

    return fun, rand_vec

fun, vec = fun_fact(50)
res = solver.optimize(fun, generations=1000, culling=75)
"""

