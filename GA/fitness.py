import numpy as np
from utils import upscale_image


def calculate_fitness(individual : np.ndarray, target_image : np.ndarray, factor : int):
    '''
    Compute the fitness of an individual by comparing it to a target image, using MSE
    Parameters:
    individual: the individual to evaluate
    target_image: the target image to compare to
    factor: the upscaling factor
    return: the fitness of the individual
    '''
    upscaled_individual = upscale_image(individual, factor)
    # Calculate MSE between the upscaled individual and the target image
    mse = np.mean((upscaled_individual.astype(np.float64) - target_image.astype(np.float64)) ** 2)
    fitness = mse

    return fitness
