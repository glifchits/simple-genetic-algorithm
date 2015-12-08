"""

    genetic
    =======

    A library implementing objective function optimization using a simple
    genetic algorithm.

    Author: George Lifchits
    Date: December 7, 2015

"""

import random


MIN = 'MINIMIZE'
MAX = 'MAXIMIZE'


def coin(probability):
    """
    :param probability: {float} 0 ≤ probability ≤ 1
    :returns: {bool} True with probability `probability`
    """
    return random.random() > probability


class Genetic:

    MIN = MIN
    MAX = MAX

    def __init__(self, obj_fun, min_or_max, p_crossover, p_mutation,
            pop_size=200, member_length=10):
        self.objfun = obj_fun
        self.minmax = min_or_max
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.population = self._generate_sample_population(pop_size,
                member_length)
        self.best_individual = None

    def _random_string(self, length):
        # generates random sequence of booleans (coin toss distribution)
        seq = (coin(0.5) for i in range(length))
        # converts boolean to int (False = 0, True = 1) and joins in a string
        return ''.join(str(int(x)) for x in seq)

    def _generate_sample_population(self, size, member_length):
        """
        :param size: {int} size of the population to generate
        :param member_length: {int} length of each member (members are strings)
        :returns: {List[str]}
        """
        return [self._random_string(member_length) for _ in range(size)]

    def reproduction(self):
        """
        Performs reproduction on the current population
        :returns: {List[str]} new population
        """
        def compute_weight(member):
            """
            :returns: {number} unnormalized roulette weight of the given member
            """
            fitness = self.objfun(member)
            if self.minmax == MIN:
                return 1/(1 + fitness)
            elif self.minmax == MAX:
                return fitness

        # compute weights for each member
        members_weights = [(m, compute_weight(m)) for m in self.population]
        # find the sum of all weights
        sum_weights = sum(w for m, w in members_weights)
        # probability density of each member is its normalized weight
        pdf = [(m, w/sum_weights) for m, w in members_weights]

        # generate new population
        new_population = []

        for i in range(len(self.population)):
            # each iteration selects one member
            rand = random.random()
            cumul = 0
            for member, end_interval in pdf:
                cumul += end_interval
                if rand <= cumul:
                    new_population.append(member)
                    break

        return new_population

    def _individual_crossover(self, parent1, parent2):
        """
        :param parent1: {string}
        :param parent2: {string}
        :returns: {(str, str)}
        """
        index = random.randint(1, len(parent1) - 1)
        head1, tail1 = parent1[:index], parent1[index:]
        head2, tail2 = parent2[:index], parent2[index:]
        return head1+tail2, head2+tail1

    def crossover(self):
        """
        Performs crossover on the current population.
        """
        new_population = []
        while len(self.population) > 2:
            if coin(self.p_crossover): # do crossover
                # pop two parents
                p1 = self.population.pop()
                p2 = self.population.pop()
                # crossover
                m1, m2 = self._individual_crossover(p1, p2)
                # add children to new population
                new_population.append(m1)
                new_population.append(m2)
            else:
                # skip this member and go to the next
                new_population.append(self.population.pop())
        # empty the current population
        while len(self.population) > 0:
            new_population.append(self.population.pop())
        # set population to new_population
        self.population = new_population

    def _individual_mutation(self, member):
        # function which returns the opposite of the given bit
        flipped = lambda x: '1' if x is '0' else '0'
        # list of chars, flipped with probability `self.p_mutation`
        chars = (flipped(c) if coin(self.p_mutation) else c for c in member)
        # return list of chars as a single string
        return ''.join(chars)

    def mutation(self):
        """
        Mutates members in the current population.
        """
        new_pop = [self._individual_mutation(m) for m in self.population]
        self.population = new_pop

    def evolve(self):
        """
        Performs iteration of SGA and maintains the best individual found.
        """
        # SGA steps
        self.reproduction()
        self.crossover()
        self.mutation()
        # now, maintain best individual
        # get the individuals in the population
        individuals = (m for m in self.population)
        # `opt` is a Python built-in function which finds optimum in a sequence
        opt = min if self.minmax == MIN else max
        # `optimum` is a function that calls `opt` with criteria objective func
        optimum = lambda x: opt(x, key=self.objfun)
        best_in_population = optimum(individuals)
        # initialize best_inidividual if necessary
        if self.best_individual is None:
            self.best_individual = best_in_population
        # find and save the overall best individual
        best = optimum([self.best_individual, best_in_population])
        self.best_individual = best


