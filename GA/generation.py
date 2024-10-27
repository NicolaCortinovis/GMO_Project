import numpy as np 

def initialize_population(pop_size : int, width : int , height : int) -> list:
    '''
    Initialize the population with random individuals
    Parameters:
    pop_size: the population size
    width: the width of the individual
    height: the height of the individual
    return: the initial population
    '''
    population = []
    for _ in range(pop_size):
        individual = generate_individual(width, height)
        population.append(individual)
    return population

def generate_individual(width : int, height : int) -> np.ndarray:
    '''
    Generate a random individual
    Parameters:
    width: the width of the individual
    height: the height of the individual
    return: the generated individual
    '''
    individual = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return individual
