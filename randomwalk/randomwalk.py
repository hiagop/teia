#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Chromosome:

    genes = []
    fitness = 0

    def __init__(self, size, alphabet):
        self.genes = np.random.choice(alphabet, size) 

    def __str__(self):
        return "{0}".format(self.genes)

    def _fitness(self, target):
        hits = 0
        for i, j in zip(self.genes, target):
            if i == j:
                hits += 1
        self.fitness = (hits*100)/len(target)

class Population:

    def __init__(self, psize, csize, alphabet):
        self.alphabet = alphabet
        self.psize = psize
        self.csize = csize
        self.population = self._gen_pop(psize, csize, alphabet)
        
    def __str__(self):
        return "\n".join(map(str,self.population))

    def sort(self):
        self.population = sorted(self.population, key=lambda chromosome: chromosome.fitness)

    def renew(self):
        index = int(len(self.population)/2)
        self.population = self._gen_pop(index, self.csize, self.alphabet) + self.population[index:]
    
    def _gen_pop(self, psize, csize, alphabet):
        pop = []
        for _ in range(psize):
            pop.append(Chromosome(csize, self.alphabet))
        return pop

class Randomwalk(Population):
    
    best_scores = []
    
    def __init__(self, psize, alphabet, target, gcounter, tcounter = 5):
        self.alphabet = alphabet
        self.target = target
        self.psize = psize
        self.csize = len(self.target)
        self.gcounter = gcounter
        self.tcounter = tcounter 
        self.population = self._gen_pop(self.psize, self.csize, self.alphabet)
        self.update(self.target)

    def renew(self):
        index = int(len(self.population)/2)
        self.population = self._gen_pop(index, self.csize, self.alphabet) + self.population[index:]
        self.update(self.target)

    def update(self, target):
        for i in range(len(self.population)):
            self.population[i]._fitness(self.target)    

    def get_fitness(self):
        return [self.population[i].fitness for i in range(len(self.population))]

    def run(self):
        for _ in range(self.tcounter):
            
            scores = []
            
            for _ in range(self.gcounter):
                self.sort()
                self.renew()
                self.sort()
                
                scores.append(self.get_fitness()[-1])
            
            self.best_scores.append(scores)
