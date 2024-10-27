import random
import numpy as np


def selection(population : list, fitnesses : list, tournament_size : int) -> list:
    '''
    Tournament selection operator, selects the best individual from a random subset of the population
    Parameters:
    population: the population
    fitnesses: the fitnesses of the population
    tournament_size: the size of the tournament
    return: the selected individuals
    '''
    selected = []
    for _ in range(len(population)):
        
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        win_index = tournament_indices[np.argmin(tournament_fitnesses)]
        selected.append(population[win_index])

    return selected

def crossover(parent1 : np.ndarray, parent2: np.ndarray) -> np.ndarray:
    '''
    Submatrices crossover operator, swaps a random submatrix region between two parents
    Parameters:
    parent1: the first parent
    parent2: the second parent
    return: the offspring
    '''
    height, width, _ = parent1.shape
    
    offspring = parent1.copy()

    # Randomly choose the top-left corner of submatrix
    x_start = random.randint(0, height - 1)
    y_start = random.randint(0, width - 1)

    # Randomly choose the size of the submatrix
    rect_height = random.randint(1, height - x_start)
    rect_width = random.randint(1, width - y_start)


    # Copy the submatrix from parent2 to the offspring (in reality it is a subtensor (?) since we're copying all channels, so three submatrices)
    offspring[x_start:x_start + rect_height, y_start:y_start + rect_width, :] = parent2[x_start:x_start + rect_height, y_start:y_start + rect_width, :]

    return offspring


# Mutation function, slightly changes the color of a random pixel (a tentative geometric mutation)
def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    '''
    Mutate an individual by changing a random pixel's color slightly.
    Parameters:
    individual: the individual to mutate
    mutation_rate: the current mutation rate
    returns: the mutated individual
    '''
    height, width = individual.shape[:2]
    num_mutations = max(1, int(height * width * mutation_rate))
    for _ in range(num_mutations):
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        # Slightly adjust the color values of the pixel by adding a random value between -20 and 20 to each channel
        # and clip the values to the [0, 255] range
        individual[x, y] = np.clip(
            individual[x, y] + np.random.randint(-20, 21, size=3),
            0, 255
        ).astype(np.uint8)
    return individual


def one_fifth_mutation_rate_update(total_mutations : int,
                                   successful_mutations : int, 
                                   mutation_rate : float,
                                   c : float,
                                   min_mutation_rate : float, 
                                   max_mutation_rate : float) -> float:
    '''
    Update the mutation rate based on the 1/5 rule (for discrete spaces) (ref: https://link.springer.com/article/10.1007/s00453-021-00854-3):
    - If the success rate is greater than 0.2, decrease the mutation rate
    - If the success rate is less than 0.2, increase the mutation rate
    - If the success rate is 0.2, do not update the mutation rate
    Parameters:
    total_mutations: the total number of mutations
    successful_mutations: the number of successful mutations
    mutation_rate: the current mutation rate
    c: the update factor
    min_mutation_rate: the minimum mutation rate
    max_mutation_rate: the maximum mutation rate
    return : the updated mutation rate
    '''
    # Compute the success rate
    if total_mutations > 0:
        p_s = successful_mutations / total_mutations

        # Update mutation rate based on the 1/5 success rule
        if p_s > 0.2:
            # Decrease mutation rate
            mutation_rate /= c  # c > 1, reduces mutation rate
        elif p_s < 0.2:
            # Increase mutation rate
            mutation_rate *= c**(0.25)  #  c > 1, increases mutation rate

        # If p_s == 0.2, do not update mutation_rate

        # Ensure mutation_rate stays within bounds
        mutation_rate = max(min_mutation_rate, min(mutation_rate, max_mutation_rate))

    return mutation_rate
