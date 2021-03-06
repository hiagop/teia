#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

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
        self.best_scores = []
        
        for _ in range(self.tcounter):
            
            self.scores = []
            
            for _ in range(self.gcounter):
                self.sort()
                self.renew()
                self.sort()
                
                self.scores.append(self.get_fitness()[-1])
            
            self.best_scores.append(self.scores)
            
    def plot(self):
        self.bs = np.array(self.best_scores)
        self.m = np.mean(self.bs, axis=0) 
        self.std = np.std(self.bs, axis=0)
        self.idx = np.arange(1, self.gcounter, 10)
        
        
        #print(len(self.bs), len(self.best_scores), len(self.m),len(self.std),len(self.idx))
        
        pl.figure(figsize=(10, 5))
        pl.title(u"Média de Acertos por Geração")
        pl.xlabel(u"Gerações")
        pl.ylabel('Acertos')
        pl.grid(alpha=0.3)
        pl.errorbar(self.idx, self.m[::10], ls=None, marker='.')
        pl.bar(self.idx, self.m[::10], yerr=self.std[::10])
        pl.show()