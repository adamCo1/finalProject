import random
import numpy as np
import pandas as pd
import sklearn.metrics
import pyitlib as metrics
import algorithms as algs
from operator import itemgetter

import data_loader as loader

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)

'''
@:param dataset: a pandas DataFrame
@:param feature_names: a list of features names
'''
class Mutual_Information_Estimator():

    def __init__(self, dataset, target, features_names):
        self.dataset = dataset
        self.features_vectors = dataset
        self.class_vector = target
        self.features_names = features_names

    '''
    @:param features_names the features to evaluate
    
    calculate the score of the selected features
    
    @:returns score of given features
    @:type real number
    '''
    def calculate_score(self, features_names):
        total_score = 0
        for feature in features_names:
            total_score = total_score + sklearn.metrics.mutual_info_score(
                self.class_vector, self.features_vectors[feature], contingency=None)
        return total_score

'''
@Params 
@:param estimator: an instance of an estimating class for use as fitness function e.g Mutual Information
@:param num_of_generations the number of maximum generations
@:param num_of_chromosomes number of chromosomes to be used, which is the size of the population
@:param num_best_chromosomes number of best chromosomes to be kept for the next generation
@:param num_rand_chromosomes number of random chromosomes from current population to be kept for next population
@:param num_crossover_children number of chromosomes to be made by crossover
@:param features_names list of the names of the features
@:param operator_probability the chance for mutation to occur 

an implementation for genetic algorithm . the class initializes an initial population, then evaluates each 
individual (chromosome) with a fitness function (estimator), then selects the best individuals to be kept for the
next generation and applies operators such as crossover, duplication and mutation.

'''
class GeneticSelector():
    def __init__(self, estimator, num_of_generations, num_of_chromosomes,  num_best_chromosomes, num_rand_chromosomes,
                 num_crossover_children, features_names, operator_probability, class_vector):
        self.estimator = estimator
        self.features_names = features_names
        self.num_of_generations = num_of_generations
        self.num_of_chromosomes = num_of_chromosomes
        self.num_best_chromosomes = num_best_chromosomes
        self.num_rand_chromosomes = num_rand_chromosomes
        self.num_crossover_children = num_crossover_children
        self.operator_probability = operator_probability
        self.best_features = []
        self.__check_parameters__()
        self.class_vector = class_vector

    def __check_parameters__(self):
        if(self.num_best_chromosomes > self.num_of_chromosomes or self.num_rand_chromosomes > self.num_of_chromosomes or
        self.num_crossover_children > self.num_of_chromosomes):
            raise ValueError('total number of chromosomes must be the highest value')

    '''
    @:param data_vector matrix of the data 
    @:param target_vector vector of the classification information

    handle the lifecycle of the algorithm

    @:returns self
    @:type class type
    '''

    def evolve(self, data_vector, target_vector):
        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []
        self.dataset = data_vector, target_vector  # set the complete dataset
        self.n_features = data_vector.shape[1]  # number of the given dataset's features
        self.best_features = []
        population = self.__initialize_population__()
        for i in range(self.num_of_generations):
            population = self.__generation_life_cycle__(population)

        self.__get_best_names__(population);
        return self

    '''
    a check to ensure the population will get to its original size after each generation
    '''
    def __checkPopulationSize__(self):
        best_to_random = (self.num_best_chromosomes + self.num_rand_chromosomes) / 2
        if(int(best_to_random * self.num_crossover_children) != self.num_of_chromosomes) :
            raise ValueError('population not stable')

    '''
        initialize a randomized population to begin the algorithm from
        
        @:return a full randomized poplation   
        @:type list
    '''
    def __initialize_population__(self):
        population = []
        randomization_factor = 0.3
        for index in range(self.num_of_chromosomes):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < randomization_factor # randomized vector at size of features
            chromosome[mask] = False
            population.append(chromosome)
        return population

    '''
        @:param population the current population of current generation 
        
        evaluate each chromose in the population by using the estimator instance given to the class.
        
        @:returns
        @:var scores sorted scores by chromoses index in the population e.g 2:20 - chromose 2 got score 20
        @:type list of lists
        
    '''
    def __fitness__(self, population):
         X, y = self.dataset
         scores = []
         for chromosome_index, chromosome in enumerate(population):
            columns_mask = []

            chromosome_names = self.__get_names__(chromosome)

            for index, value in enumerate(chromosome): # get the names for the chromose mask
                if value == True:
                    columns_mask.append(self.features_names[index])
            data_to_fit = X[columns_mask] # send only features with True in the chromosome
            score = self.estimator.calculate_score(columns_mask)
            scores.append([chromosome_index,score])
         scores.sort(key=itemgetter(1), reverse=True) # sort the scores by highest score to lowest

         return scores

    '''
    @:param scores_sorted list of chromosomes scores after fitness
    @:param population current population
    
    select the best chromosomes from current generation to the next generation
    select random chromosomes for the next generation
    
    @:returns 
    @:type list
    '''
    def __select_best_chromosomes__(self, scores_sorted, population):
        next_population = []
        chromosome_index = 0
        for chromosome_num in range(self.num_best_chromosomes):# get the best chromosomes for next generation
            next_population.append(population[scores_sorted[chromosome_num][chromosome_index]])
        for i in range(self.num_rand_chromosomes): # pick random chromosomes for next generation
            next_population.append(random.choice(population))
        return next_population

    '''
    @:param population current population
    
    fill the next generation by creating new "children" from the chosen chromosomes of the last 
    generation
    
    @:returns population after crossovers
    @:type list
    '''
    def __crossover__(self, population):
        next_population = population
        while(len(next_population) < self.num_of_chromosomes):
            first_parent_random_index = random.randint(self.num_best_chromosomes, len(population)-1)
            second_parent_random_index = random.randint(self.num_best_chromosomes, len(population)-1)
            # choose 2 parents at random
            chromosome1, chromosome2 = population[first_parent_random_index], population[second_parent_random_index]
            child = chromosome1
            mask = np.random.rand(len(child)) > 0.5
            child[mask] = chromosome2[mask]
            next_population.append(child)
        return next_population

    '''
    @:param population current population
    
    mutate chromosomes in some probability given to the class
    
    @:returns population after mutation step
    @:type list
    '''
    def __mutate__(self, population):
        next_population = []
        for index in range(len(population)):
            chromosome = population[index]
            if(index > self.num_best_chromosomes): # dont mutate the best
                if(self.__should_apply_operator__()):
                    mask = np.random.rand(len(chromosome)) < 0.3
                    chromosome[mask] = True
            next_population.append(chromosome)
        return next_population

    '''
    '''
    def __generation_life_cycle__(self, population):
        # Selection, crossover and mutation
        sorted_scores = self.__fitness__(population)
        population = self.__select_best_chromosomes__(sorted_scores, population)
        population = self.__crossover__(population)
        population = self.__mutate__(population)
        self.scores_best.append(sorted_scores[0])
        print('best scores of current generation : ')
        print(sorted_scores[0][1])
        return population

    '''
    @:param best_population the best chromosomes at the end of the algorithm
    
    sets the best features chosen by their names
    '''
    def __get_best_names__(self, best_population):
        for index, chromosome in enumerate(best_population):
            names = []
            for feature_index, feature in enumerate(chromosome): # get the names from each chromosome
                if feature == True:
                    names.append(self.features_names[feature_index])
            if names not in self.best_features: # no duplicates
                self.best_features.append(names)

    def __get_names__(self, chromosome):
        names = []
        for feature_index, feature in enumerate(chromosome):  # get the names from each chromosome
            if feature == True:
                names.append(self.features_names[feature_index])
        return names

    '''
    @:returns apply the operator or not
    @:type boolean
    '''
    def __should_apply_operator__(self):
        return random.random() < self.operator_probability

def main():
    process = loader.data_processing('wdbc.csv', sep=',', number_of_bins=10)

    features_vectors, class_vector, features_names, class_name = process.prepare_data()

    # target_vector = features_vectors[features_vectors.columns[-1]]

    features_vectors = algs.fcbf(features_vectors, class_vector, 0.1)
    features_names = features_vectors.columns.values

    for i in range(20):
        mu = Mutual_Information_Estimator(features_vectors, class_vector, features_names)
        selector = GeneticSelector(estimator = mu,
                                   num_of_generations = 5,
                                   num_of_chromosomes = 30,
                                   num_best_chromosomes = 10,
                                   num_rand_chromosomes = 5,
                                   num_crossover_children = 5,
                                   features_names = features_names,
                                   operator_probability = 0.3,
                                   class_vector = class_vector)
        selector.evolve(features_vectors, class_vector)
        print(i, selector.best_features)

if __name__ == '__main__':
    main()