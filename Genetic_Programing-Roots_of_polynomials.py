import numpy as np
import random
import copy
from prettytable import PrettyTable

np.random.seed(512)
random.seed(512)


class Chromosome:
    def __init__(self, x, f) -> None:
        self.x = x
        self.f = f

    @property
    def y(self):
        return np.abs(np.array([self.f(i) for i in self.x]))

    @property
    def fitness(self):
        return np.mean(self.y)

    def crossover(self, chr):
        point = np.random.randint(len(chr)+1)
        temp = chr[0:point].copy()
        chr[0:point] = self[0:point].copy()
        self[0:point] = temp.copy()
        return self, chr

    def mutation(self, begin, end):
        point = np.random.randint(len(self))
        self[point] = np.random.uniform(begin, end)
        return self

    def __getitem__(self, index):
        return self.x[index]

    def __setitem__(self, index, value):
        self.x[index] = value

    def __len__(self):
        return len(self.x)

    def __repr__(self) -> str:
        return str(self.x)


class GA:
    def __init__(self, p, ds, f, n_roots, crate=0.8, mrate=0.01) -> None:
        self.npopulation = p
        self.domain_search = ds
        self.f = f
        self.n_roots =n_roots
        self.crossover_rate = crate
        self.mutation_rate = mrate

    def genchromosome(self):
        ds0 = self.domain_search[0]
        ds1 = self.domain_search[1]
        ch = np.random.uniform(ds0, ds1, self.n_roots)
        return Chromosome(ch, self.f)

    def init_population(self):
        population = []
        for _ in range(self.npopulation):
            population.append(self.genchromosome())
        return population

    def roulette_wheel_select(self, population):
        weights = [1 / chr.fitness for chr in population]
        new_population = []
        for i in range(len(population)):
            new_population.append(copy.deepcopy(random.choices(population, weights)[0]))
        return new_population

    def select(self, population):
        plen = len(population)
        sorted_population = sorted(population, key=lambda x:x.fitness)
        best_population = sorted_population[0:int(0.1*plen)]
        rest_population = self.roulette_wheel_select(sorted_population[int(0.1*plen)::])
        return best_population + rest_population

    def crossover(self, population):
        i = 0
        while(i < len(population)):
            if i+1 != len(population) and random.random() < self.crossover_rate:
                population[i].crossover(population[i+1])
            i += 2
        return population

    def mutation(self, population):
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                population[i].mutation(self.domain_search[0], self.domain_search[1])
        return population

    def is_terminated(self, population, error):
        for chr in population:
            if(np.all(np.where(chr.y <= error, True, False))):
                return True
        return False

    def run(self, generation_num, error=0.001):
        i = 0
        population = self.init_population()
        while(i < generation_num and not self.is_terminated(population, error)):
            population = self.crossover(population)
            population = self.mutation(population)
            population = self.select(population)
            i += 1

        return population[0]


def f1(x):
    return 2*x - 4

def f2(x):
    return x**2-8*x+4

def f3(x):
    return 4*x**3-5*x**2+x-1

def f4(x):
    return 186*x**3-7.22*x**2+15.5*x-13.2


fs = [f1, f2, f3, f4]
na = [1, 2, 1, 1]
names = ['2x-4', 'x^2 - 8x + 4', '4x^3 - 5x^2 + x - 1', '186x^3 - 7.22x^2 + 15.5x - 13.2']

table = PrettyTable()
table.field_names = ['f', 'roots', 'error']

for f, i, name in zip(fs, na, names):
    ga = GA(250, (0, 10), f, i, crate=0.95, mrate=0.05)
    answer = ga.run(1000)
    table.add_row([name, answer, answer.y])

print(table)