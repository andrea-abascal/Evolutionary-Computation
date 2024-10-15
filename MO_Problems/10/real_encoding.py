#Encoding        Real encoding 	
#Initialization  Random 	
#Crossover 	     Simulated Binary Crossover (SBX) 	Pc = 0.9; nc = 20
#Mutation 	     Parameter-based Mutation (PM)	    Pm = 1/n; nm = 20
#Selection 	     Binary tournament (deterministic)

import numpy as np
import matplotlib.pyplot as plt
import json, time

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

# Randomly initialize population
def initialize_population(pop_size, bounds):
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(pop_size, bounds.shape[0]))
    return population

# Fitness calculation
def fitness_function(population):
    return np.array([objective_function(ind) for ind in population])

# Selection using binary tournament selection
def binary_tournament_selection(population, fitness):
    selected = []
    for _ in range(len(population) // 2): 
        idx1, idx2 = np.random.choice(range(len(population)), size=2, replace=False)
        if fitness[idx1] < fitness[idx2]:
            selected.append(population[idx1])
        else:
            selected.append(population[idx2])
    return np.array(selected)

# Simulated Binary Crossover
def crossover(parents, crossover_rate, nc = 20):
    offspring = parents.copy()
    if  np.random.rand() < crossover_rate:
        for i in range(0, len(parents),2):
            p1 = parents[i]
            p2 = parents[i + 1]
            u = np.random.rand()
            if u <= 0.5:
                B =  (2*u)**1/(nc+1)
            else:
                B =  (1/(2*(1-u)))**1/(nc+1)
            c1 = 0.5*abs((p1+p2) - B*(abs(p2-p1)))
            c2 = 0.5*abs((p1+p2) + B*(abs(p2-p1)))
            offspring[i]= c1
            offspring[i+1]= c2
    return offspring

# Parameter-based mutation
def mutation(offspring, mutation_rate,bounds, nm = 20):
    y_l = bounds[0][0]
    y_u = bounds[0][1]
    
    for i in range(len(offspring)):
        ind = offspring[i]
        for var in range(len(ind)):
            if np.random.rand() < mutation_rate:
                y = ind[var]
                u = np.random.rand()
                max_delta = y_u-y_l
                d = np.min([y - y_l, y_u - y])/ max_delta
                if u <= 0.5:
                    dq =  ((2*u + (1 - 2*u)*(1 - d)**(nm +1))**(1/(nm + 1))) - 1
                else:
                    dq = 1 - (2*(1 - u)+ 2*(u -0.5)*(1 - d)**(nm +1))**(1/(nm + 1))
                mut_ind = y + dq*max_delta
                ind[var]= mut_ind
                offspring[i] = ind
    return offspring

# Genetic Algorithm
def genetic_algorithm(objective_function, bounds, mutation_rate, crossover_rate, pop_size=100, num_generations=500 ):
    population = initialize_population(pop_size, bounds)
    best_fitness_list = []
    best_individual_list = []
    start = time.time()
    for generation in range(num_generations):
        fitness = fitness_function(population)
        parents = binary_tournament_selection(population, fitness)
        offspring_crossover = crossover(parents,  crossover_rate)
        offspring_mutation = mutation(offspring_crossover, mutation_rate, bounds)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation

        # Track best solution
        best_fitness_idx = np.argmin(fitness)
        best_individual = population[best_fitness_idx]
        best_fitness = fitness[best_fitness_idx]
        best_fitness_list.append(best_fitness)
        best_individual_list.append(best_individual.tolist())
    end = time.time()
    t = end-start
    #plt.plot(best_fitness_list)
    #plt.xlabel('Generation')
    #plt.ylabel('Best Fitness')
    #plt.title('Fitness Over Generations')
    #plt.show()
    gen = np.argmin(best_fitness_list)
    
    return best_individual, best_fitness, best_fitness_list ,best_individual_list, gen, t

data = {}
folder = '/home/andrea/MCC/Repos/Evolutionary Computation/MO_Problems/10'

for i, test in enumerate(tests):
    fbest_values = []
    
    for n in range(1):
        print(f"--- Test Case {i+1} -- Experiment {n+1} ---")
        objective_function = test["objective_function"]
        bounds = np.array(test["bounds"])
        mutation_rate = (1/len(bounds))
        crossover_rate = 0.9

        # Run the genetic algorithm
        best_solution, best_value, fitness_list,best_individual_list, gen, t = genetic_algorithm(objective_function, bounds, mutation_rate, crossover_rate)
        fbest_values.append(best_value)
        print(f'Best value: {best_value} at solution: {best_solution} and generation: {gen}')

    # Calculate statistics for this test case
    data_array = np.array(fbest_values)
    mean_value = np.mean(data_array)
    std_dev = np.std(data_array)
    min_value = np.min(data_array)
    max_value = np.max(data_array)
    print(f"\n--- Resume for Test Case {i+1} ---")
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}\n")


    data[f'test_case{i+1}_time'] = t
    data[f'test_case{i+1}_fitness'] = fitness_list
    data[f'test_case{i+1}_path'] = best_individual_list
    data[f'test_case{i+1}_sol'] = best_solution.tolist()


    with open(f"{folder}/real_encoding.json", "w") as final:
        json.dump(data, final)

