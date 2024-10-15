# Evolutionary Algorithm example
import random
POP_SIZE = 500                        # Number of chromosomes in the list
MUT_RATE = 0.1                        # Rate of string mutation
TARGET = 'andrea abascal'
GENES = ' abcdefghijklmnopqrstuvwxyz' # Options from which the population would be created

# STEP 1 - INITIALIZATION
# Generating a population of size equal to target string. Each string is called a chromosome
# and each chromosome consists of only the letters defined in genes.

def initialize_P0(TARGET):
    population = []
    target_len = len(TARGET)

    for i in range(POP_SIZE):
        temp = []
        for j in range(target_len):
            temp.append(random.choice(GENES))
        population.append(temp)
    return population

# STEP 2 - FITNESS EVALUATION
# Fitness is computing by comparing the number of letters matching the target.
# A larger fitness means it is far from the target, a fitness of 0 means the target was found.
def fitness_eval(TARGET, chrom_from_pop):
    difference = 0
    for target_char, chromosome_char in zip(TARGET, chrom_from_pop):
        if target_char != chromosome_char:
            difference += 1
    return [chrom_from_pop, difference]

# STEP 3 - SELECTION
# Select best chromosomes by sorting them in ascending fitness order
# only the top 50% of parents are selected to avoid bad chromosomes in the future
def selection(population, TARGET):
  sorted_chrom_pop = sorted(population, key= lambda x: x[1])
  return sorted_chrom_pop[:int(0.5*POP_SIZE)]

# STEP 4 - GENETIC OPERATORS (CROSSOVER)
# A process that adds diveristy to the population in chich one parent is randomly
# selected from the best chromosomes and another parent is selected from initial population.
# The crossover defines this point where information is swapped between parents.
def crossover(selected_chrom, CHROMO_LEN, population):
  offspring_cross = []
  for i in range(int(POP_SIZE)):
    parent1 = random.choice(selected_chrom)
    parent2 = random.choice(population[:int(POP_SIZE*50)])

    p1 = parent1[0]
    p2 = parent2[0]

    crossover_point = random.randint(1, CHROMO_LEN-1)
    child =  p1[:crossover_point] + p2[crossover_point:]
    offspring_cross.extend([child])
  return offspring_cross

# STEP 4 - GENETIC OPERATORS (MUTATION)
# Randomly select a letter from each chromosome and replace it with another letter present in genes.
# The replacement probability depends on the MUT_RATE and the random number generated.
def mutate(offspring, MUT_RATE):
   mutated_offspring = []

   for arr in offspring:
    for i in range(len(arr)):
        if random.random() < MUT_RATE:
            arr[i] = random.choice(GENES)
    mutated_offspring.append(arr)
   return mutated_offspring

# STEP 5 - REPLACEMENT
def replace(new_gen, population):
  for _ in range(len(population)):
      if population[_][1] > new_gen[_][1]:
        population[_][0] = new_gen[_][0]
        population[_][1] = new_gen[_][1]
  return population

def main(POP_SIZE, MUT_RATE, TARGET, GENES):
   init_population = initialize_P0(TARGET)
   found = False
   population = []
   generation = 1

   for i in range(len(init_population)):
       population.append(fitness_eval(TARGET, init_population[i]))

   while not found:
      selected = selection(population, TARGET)
      population = sorted(population, key= lambda x: x[1])
      crossovered = crossover(selected, len(TARGET), population)
      mutated = mutate(crossovered, MUT_RATE)

      new_gen = []
      for i in mutated:
        new_gen.append(fitness_eval(TARGET,i))
      population = replace(new_gen, population)

      if (population[0][1] == 0):
        print('Target found')
        print('String: ' + str(population[0][0]) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0][1]))
        break
      print('String: ' + str(population[0][0]) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0][1]))
      generation+=1

main(POP_SIZE, MUT_RATE, TARGET, GENES)



