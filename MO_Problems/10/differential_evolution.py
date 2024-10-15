#Encoding        Real encoding 	
#Initialization  Random 	
#Crossover 	     Simulated Binary Crossover (SBX) 	Pc = 0.9; nc = 20
#Mutation 	     Parameter-based Mutation (PM)	    Pm = 1/n; nm = 20
#Selection 	     Binary tournament (deterministic)

import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Test cases
tests = [{
        "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
    },
    {
        "objective_function": lambda x: 20 + sum([(x[i]**2 - 10 * np.cos(2 * np.pi * x[i])) for i in range(2)]),
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
    }
]
def plot_contour(f, solution, bounds, path, filename):
    """Plot contour of the function and the solution path."""
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f([x1, x2]) for x1 in X1[i]] for i, x2 in enumerate(x2)])
    x = np.linspace(-1, 1, 400)


    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=np.linspace(np.min(Z), np.max(Z), 30), cmap='viridis')
    plt.colorbar(label='Objective function value')

    # Plot the path of iterations
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'r-o', label='Real Encoding GA')
    plt.scatter(solution[0], solution[1], color='blue', s=100, zorder=5, label=f'Solution at {solution}')
    
    plt.title(f'Contour Plot for 500 Generations')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.show()

# Randomly initialize population
def initialize_population(pop_size, bounds):
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(pop_size, bounds.shape[0]))
    return population

# Random selection
def random_idx_selection(population):
    selection_dict = {}

    for i in range(len(population)):
        # Create a list of all possible indexes excluding the current individual index
        possible_indexes = list(range(len(population)))
        possible_indexes.remove(i)
        
        # Randomly select 3 indexes that are not the same as the current individual's index
        selected_indexes = np.random.choice(possible_indexes, size=3, replace=False)
        
        # Store the selected indexes for the individual
        selection_dict[i] = selected_indexes.tolist()
    
    return selection_dict


def crossover(trial_vectors, target_vectors,cr):
    offspring_vectors = []
    
    for i in range(len(trial_vectors)):
        target_vector = target_vectors[i]
        trial_vector = trial_vectors[i]
        D = len(target_vector)  # Dimension of the vector
        j_rand = np.random.randint(0, D)  # Random index to ensure at least one trial component is copied
        
        # Create the offspring vector
        offspring = []
        for j in range(D):
            rand_j = np.random.rand()  # Generate a random number between 0 and 1
            if rand_j <= cr or j == j_rand:
                offspring.append(trial_vector[j])  # Take from trial vector
            else:
                offspring.append(target_vector[j])  # Take from target vector
        
        offspring_vectors.append(np.array(offspring))
    
    return offspring_vectors

# Parameter-based mutation
def mutation(pop, parents, bounds, F):
    trial_vectors = []
    target_vectors = []
    for i in range(len(parents)):
        idxs = parents[i]
        target_vector = pop[idxs[0]]
        trial_vector = target_vector + F * (pop[idxs[1]]- pop[idxs[2]])
        trial_vector = [np.clip(trial_vector[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
        target_vectors.append(np.array(target_vector))
        trial_vectors.append(trial_vector)
    return trial_vectors, target_vectors

def selection(pop, offspring):
    new_population = []
    fitnesses = []
    for i in range(len(pop)):
        # Evaluate the objective function for both the target and trial vectors
        target_fitness = objective_function(pop[i])
        trial_fitness = objective_function(offspring[i])
        
        # Select the vector with the better objective function value
        if trial_fitness < target_fitness:  # Minimization problem
            new_population.append(offspring[i])
            fitnesses.append(trial_fitness)
        else:
            new_population.append(pop[i])
            fitnesses.append(target_fitness)
    
    return new_population,fitnesses

# Genetic Algorithm
def genetic_algorithm(objective_function, bounds, F, crossover_rate, pop_size=100, num_generations=500 ):
    population = initialize_population(pop_size, bounds)
    best_fitness_list = []
    best_individual_list = []

    start = time.time()
    for generation in range(num_generations):
        parents = random_idx_selection(population)
        trial_vectors, target_vectors = mutation(population,parents,bounds,F)
        offspring_crossover = crossover(trial_vectors,target_vectors,crossover_rate)
        population,fitness = selection(population, offspring_crossover)

        # Track best solution
        best_fitness_idx = np.argmin(fitness)
        best_individual = population[best_fitness_idx]
        best_fitness = fitness[best_fitness_idx]
        best_fitness_list.append(best_fitness)
        best_individual_list.append(best_individual.tolist())
    end = time.time()
    t = end-start
    

    plt.plot(best_fitness_list)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Over Generations')
    plt.show()
    gen = np.argmin(best_fitness_list)
    
    return best_individual, best_fitness, best_fitness_list , gen, best_individual_list, t

data = {}
folder = '/home/andrea/MCC/Repos/Evolutionary Computation/MO_Problems/10'
for i, test in enumerate(tests):
    fbest_values = []
    
    for n in range(1):
        print(f"--- Test Case {i+1} -- Experiment {n+1} ---")
        objective_function = test["objective_function"]
        bounds = np.array(test["bounds"])
        F = 0.5
        crossover_rate = 0.6

        # Run the genetic algorithm
        best_solution, best_value, fitness_list, gen, best_individual_list,t = genetic_algorithm(objective_function, bounds, F, crossover_rate)
        fbest_values.append(best_value)
        print(f'Best value: {best_value} at solution: {best_solution} and generation: {gen}')
        plot_contour(objective_function,best_solution,bounds, best_individual_list, f'{folder}/contour_problem_{i+1}.png')
        

    # Calculate statistics for this test case
    data_array = np.array(fbest_values)
    mean_value = np.mean(data_array)
    std_dev = np.std(data_array)
    min_value = np.min(data_array)
    max_value = np.max(data_array)
    print(f"\n--- Resume for Test Case {i+1} ---")
    print(f'Total time: {t} seconds')
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}\n")
    data[f'test_case{i+1}_time'] = t
    data[f'test_case{i+1}_fitness'] = fitness_list
    data[f'test_case{i+1}_path'] = best_individual_list
    data[f'test_case{i+1}_sol'] = best_solution.tolist()


    with open(f"{folder}/practice_10.json", "w") as final:
        json.dump(data, final)