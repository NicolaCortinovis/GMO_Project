import numpy as np 
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


from operators import selection, crossover, mutate, one_fifth_mutation_rate_update
from fitness import calculate_fitness
from utils import load_image, downsample_image, upscale_image
from generation import initialize_population


def genetic_algorithm(image_path : str,
                      factor : int =8,
                      pop_size : int =100,
                      generations : int = 1000,
                      mutation_rate : float = 0.05,
                      tournament_size : int = 5,
                      replacement_size : int = 30,
                      c : int = 1.1,
                      min_mut_rate : float = 0.001,
                      max_mut_rate : float = 0.1) -> np.ndarray:
    '''
    Genetic algorithm that given an image path, generates a low-resolution version of the image
    using rgb colors and a chunk size of 8x8 pixels
    Parameters:
    image_path: the path to the image
    factor: the downscaling factor
    pop_size: the population size
    generations: the number of generations
    mutation_rate: the initial mutation rate
    tournament_size: the tournament size for selection
    replacement_size: the number of individuals to replace in the population (steady state approach)
    c : the update factor for the mutation rate (constant and > 1)
    min_mut_rate: the minimum mutation rate
    max_mut_rate: the maximum mutation rate
    return: the low-resolution image upscaled to the original size
    '''

    if(c <= 1):
        raise ValueError('This hyperparameter governs the mutation rate update based on the 1/5 rule and must be <= 1')

    # Load the image and downsample it
    original_image = load_image(image_path)
    downsampled_target = downsample_image(original_image, factor)

    # Initialize the population and the best fitness history
    height, width, _ = downsampled_target.shape
    population = initialize_population(pop_size, width, height)
    best_fitness_history = []

    for generation in range(generations):
        
        # Parallel fitness calculation for the first generation
        if(len(best_fitness_history) == 0):
            with ThreadPoolExecutor() as executor:
                fitnesses = list(executor.map(
                    lambda ind: calculate_fitness(ind, original_image, factor),
                    population
                ))

        # Record the best fitness
        best_fitness = min(fitnesses)
        best_fitness_history.append(best_fitness)
        best_individual = population[np.argmin(fitnesses)]


        # Select the possible parents
        selected = selection(population, fitnesses, tournament_size)

        # Create next generation
        new_offspring = []
        new_offspring_fitnesses = []

        # Used for the 1/5 rule update of the mutation rate
        successful_mutations = 0
        total_mutations = 0


        # Generate the new population
        for individual in range(pop_size):
            parent1, parent2 = random.sample(selected, 2)
            offspring = crossover(parent1, parent2)
            new_offspring.append(offspring)

        # Compute fitnesses before mutation in parallel
        with ThreadPoolExecutor() as executor:
            fitnesses_before = list(executor.map(
                lambda ind: calculate_fitness(ind, original_image, factor),
                new_offspring
            ))
        
        # Apply mutation to all new offspring
        new_offspring_mutated = [mutate(ind, mutation_rate) for ind in new_offspring]


        # Compute fitnesses after mutation in parallel
        with ThreadPoolExecutor() as executor:
            fitnesses_after = list(executor.map(
                lambda ind: calculate_fitness(ind, original_image, factor),
                new_offspring_mutated
            ))

        # Determine the number of successful mutations
        total_mutations = pop_size
        successful_mutations = sum(1 for before, after in zip(fitnesses_before, fitnesses_after) if after < before)

        # Save the new offspring and their fitnesses
        new_offspring = new_offspring_mutated
        new_offspring_fitnesses = fitnesses_after

        # Identify the n best offspring
        n_best_indices = np.argsort(new_offspring_fitnesses)[:replacement_size]
        n_best_offspring = [new_offspring[i] for i in n_best_indices]
        n_best_offspring_fitnesses = [new_offspring_fitnesses[i] for i in n_best_indices]

        # Identify the n worst individuals in the current population
        n_worst_indices = np.argsort(fitnesses)[-replacement_size:]

        # Replace the n worst individuals with the n best offspring (steady state approach)
        for i, idx in enumerate(n_worst_indices):
            population[idx] = n_best_offspring[i]
            fitnesses[idx] = n_best_offspring_fitnesses[i]
        
        # Note that this for cycle ensures that the fitness for the new population is already computed for the next generation


        # Update mutation rate based on the 1/5 rule
        mutation_rate = one_fifth_mutation_rate_update(total_mutations, successful_mutations, mutation_rate, c, min_mut_rate, max_mut_rate)

        # Print best fitness and plot the best individual every 100 generations
        if generation % 100 == 0 or generation == generations - 1:
            print(f'Generation {generation}, Best Fitness (MSE): {best_fitness}')
            plt.imshow(upscale_image(best_individual, factor))
            plt.title(f'Generation {generation}')
            plt.axis('off')
            plt.show()


    # Return the best individual (final low-resolution image upscaled) and the fitness history
    return upscale_image(best_individual, factor), best_fitness_history
