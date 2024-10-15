#Encoding        Binary encoding 	
#Initialization  Random 	
#Crossover 	     Single point crossover 	 Pc = 0.9
#Mutation 	     Binary mutation 	         Pm = 1/n
#Selection 	     Roulette wheel selection

import numpy as np
import matplotlib.pyplot as plt
import json, csv

# Test cases
tests = [{
        "objective_function": lambda x: 100 * (x[0]**2 - x[1])**2 + (1 -x[0])**2,
        "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
    },
    {
        "objective_function": lambda x: 20 + sum([(x[i]**2 - 10 * np.cos(2 * np.pi * x[i])) for i in range(2)]),
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
    },
  
    {
        "objective_function": lambda x: 50 + sum([(x[i]**2 - 10 * np.cos(2 * np.pi * x[i])) for i in range(5)]),
        "bounds": [(-5.12, 5.12), (-5.12, 5.12),(-5.12, 5.12), (-5.12, 5.12),(-5.12, 5.12)],
    }
]

# Bits size
def bits_size(bounds,precision):
    b = bounds[0]
    num_bits = int(np.log2((b[1]-b[0])*(10**precision))+0.99)
    return num_bits

# Binary to real-value decoding
def decoding(bounds, bits, chromosome):
    real_chromosome = list()
    for i in range(len(bounds)):
        start = i * bits
        end = (i+1) * bits
        substring = chromosome[start:end]
        # Convert binary to decimal
        decimal_value = int(''.join(map(str, substring)), 2)
        # Scale decimal to real range
        min_bound, max_bound = bounds[i]
        real_value = min_bound + (decimal_value / (2**bits - 1)) * (max_bound - min_bound)
        real_chromosome.append(real_value)
    return real_chromosome

# Initialize population (Random binary population)
def initialize_population(pop_size, num_bits, num_variables):
    return np.random.randint(2, size=(pop_size, num_bits * num_variables))

# Fitness calculation
def fitness_function(population, bounds, num_bits):
    fitness = []
    for individual in population:
        real_values = decoding(bounds,num_bits, individual)
        fitness.append(objective_function(real_values))  
    return np.array(fitness)

# Selection (roulette wheel selection)
def select_parents(population, fitness, num_parents):
    fitness_inv = 1 / (fitness)  # Inverse fitness (minimization problem)
    probs = fitness_inv / np.sum(fitness_inv)
    parents_idx = np.random.choice(np.arange(len(population)), size=num_parents, p=probs)
    return population[parents_idx]


# Crossover (single-point crossover)
def crossover(parents, offspring_size, crossover_rate):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] // 2)
    
    for k in range(offspring_size[0]):
        if np.random.rand() < crossover_rate:
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Mutation (flip bits)
def mutation(offspring, mutation_rate):
    for ind in offspring:
        for gene in range(ind.shape[0]):
            if np.random.rand() < mutation_rate:
                ind[gene] = 1 - ind[gene]  # Flip the bit
    return offspring

# Genetic Algorithm
def genetic_algorithm(objective_function, bounds, num_bits, mutation_rate,crossover_rate, pop_size=100, num_generations=500):
    num_variables = bounds.shape[0]
    population = initialize_population(pop_size,num_bits, num_variables)
    best_fitness_list = []

    for generation in range(num_generations):
        fitness = fitness_function(population, bounds, num_bits)
        parents = select_parents(population, fitness, num_parents=pop_size//2)
        offspring_crossover = crossover(parents, offspring_size=(pop_size - parents.shape[0], num_bits * num_variables),crossover_rate = crossover_rate)
        offspring_mutation = mutation(offspring_crossover, mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
       
        best_fitness_idx = np.argmin(fitness)
        best_individual = population[best_fitness_idx]
        best_fitness = fitness[best_fitness_idx]
        best_fitness_list.append(best_fitness)

    
    best_solution_real = decoding(bounds, num_bits, best_individual)
    #plt.plot(best_fitness_list)
    #plt.xlabel('Generation')
    #plt.ylabel('Best Fitness')
    #plt.title('Fitness Over Generations')
    #plt.show()
    gen = np.argmin(best_fitness_list)

    return best_solution_real, np.min(fitness), best_fitness_list, gen

data = {}
# File to store results
output_file = "binary_genetic_algorithm_results.csv"

# Open CSV file to write results
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(["Test Case", "Experiment", "Gen", "Best Value", "Mean", "Standard Deviation", "Min", "Max"])

    # Iterate over each test case
    for i, test in enumerate(tests):
        fbest_values = []
        gens = []  # Track generations for each experiment

        for n in range(20):
            print(f"--- Binary Test Problem {i+1} -- Experiment {n+1} ---")
            objective_function = test["objective_function"]
            bounds = np.array(test["bounds"])
            digits_precision = 4
            num_bits = bits_size(bounds, digits_precision)
            mutation_rate = 1 / (num_bits * 2)
            crossover_rate = 0.9

            # Run the genetic algorithm
            best_solution, best_value, fitness_list, gen = genetic_algorithm(objective_function, bounds, num_bits, mutation_rate, crossover_rate)
            print(f'Best value: {best_value} at solution: {best_solution} and generation: {gen}')
            fbest_values.append(best_value)
            gens.append(gen)

            # Write Gen and Best Value for each experiment to CSV
            writer.writerow([i+1, n+1, gen, best_value, "", "", "", ""])

        # Calculate statistics for this test case
        data_array = np.array(fbest_values)
        mean_value = np.mean(data_array)
        std_dev = np.std(data_array)
        min_value = np.min(data_array)
        max_value = np.max(data_array)

        # Write statistics to CSV after all experiments of the test case
        writer.writerow([i+1, "Summary", "", "", mean_value, std_dev, min_value, max_value])

        print(f"\n--- Resume for Test Case {i+1} ---")
        print(f"Mean: {mean_value}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Minimum: {min_value}")
        print(f"Maximum: {max_value}\n")

    #data[f'test_case{i+1}'] = fitness_list


#with open(f"binary_encoding.json", "w") as final:
#    json.dump(data, final)
    

