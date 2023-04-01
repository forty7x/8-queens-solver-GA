import time
import random
import numpy as np
import matplotlib.pyplot as plt

best_fitness_history = []
average_fitness_history = []
tournament_size = 10

def draw_board(cols):
  # draws the board with the queens on it
  # use unicode characters for white and black squares and queens
  white = "\u25A1"
  black = "\u25A0"
  white_queen = "\u2655"
  black_queen = "\u265B"
  for row in range(8): 
    line = "" 
    for col in range(8): 
      if (row + col) % 2 == 0: 
        if cols[row] == col: 
          line += white_queen + " " 
        else: 
          line += white + " " 
      else: 
        if cols[row] == col: 
          line += black_queen + " " 
        else: 
          line += black + " " 
    print(line) 

#fitness function checks if queens are attacking eachother 
def fitness(solution):
    conflicts = 0
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if solution[i] == solution[j] or abs(solution[i] - solution[j]) == j - i:
                conflicts += 1
    return 1 / (conflicts + 1)  


def crossover(parent1, parent2):
    # Create a new solution by randomly selecting half of the queens from each parent
    n = len(parent1)
    crossover_point = random.randint(1, n - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(solution):
    # Swap the positions of two randomly selected queens
    n = len(solution)
    i, j = random.sample(range(n), 2)
    solution[i], solution[j] = solution[j], solution[i]
    return solution

def genetic_algorithm(population_size, crossover_probability, mutation_probability, max_iterations):
    # Initialize the population
    #population = generate_population(population_size)
    population = []
    for i in range(population_size):
        solution = list(range(8))
        random.shuffle(solution)
        population.append(solution)
    #draw_board(solution) this was to view an initial sample from the 1st generation population

    
    for i in range(max_iterations):
        
        fitness_values = [fitness(solution) for solution in population]

        # if solution found, return it
        if 1.0 in fitness_values:
            index = fitness_values.index(1.0)
            return population[index]
        
        best_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)

        # Select parents for reproduction
        #takes a random sample from the population and returns the fitness
        parents = [max(random.sample(population, tournament_size), key=fitness) for i in range(population_size)]
        #  crossover and mutation step
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_probability:
                child1 = mutate(child1)
            if random.random() < mutation_probability:
                child2 = mutate(child2)
            offspring += [child1, child2]

        # Replace the least fit members of the population with the offspring
        population = sorted(population, key=fitness, reverse=True)
        population[-len(offspring):] = offspring

    # Return the best solution found
    return max(population, key=fitness)

solution = genetic_algorithm(population_size=1000, crossover_probability=0.9, mutation_probability=0.1, max_iterations=10000)
'''plt.plot(best_fitness_history, label='Best Fitness')
plt.plot(average_fitness_history, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Best and Average Fitness over Time')
plt.legend()
plt.show()
print(solution)'''
#list = solution
draw_board(solution)

