#Here are the imported libaries and functions that I will be using 
import time
import random
import heapq
import math
from matplotlib.figure import Figure
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict

class Particle:
    def __init__(self, n):
        self.position = [random.randint(0, 1) for x in range(n)] #represents a possible solution
        self.velocity = [random.random() for x in range(n)] #the velocity of a particle which is initially randomised
        self.pbest = self.position[:] #the personal best solution seen
        self.pbest_fitness = 0 #the value of the personal best solution
        #Used for the graph/visualisation
        self.pbest_list = [] 
        self.velocity_list = []

class BinaryKP:
    def __init__(self, weights, values, capacity, algorithm_data):
        #Constant terms between each algorithm
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.algorithm_data = algorithm_data
        #Used for easy access in the recursive algorithm
        self.recursion_counter = 0
        self.hits = 0

    #Recursivly called ploting methods
    def plot_node(self, axis, x, y, lastnode=False):
        node_size = 10
        axis.scatter(x, y, s=node_size, color='black', label='node', zorder=2)

        if lastnode:
            #If it is a last node, the branch will end and display a diamond 
            axis.scatter(x, y-0.25, s=2, color='black', marker='d', alpha=0.5, label='hidden nodes')

    def solve_01_knapsack_BF_iterative(self):
        #graph max_length
        max_length = 10000
        #Loop through all subsets/combinations of items and return one with the highest total value that doesn't exceed self.capacity
        #A perhaps interesting way of doing this is counting upwards in binary
        #In order to count up efficiently, we must use gray codes so that between each iteration we only flip one bit, therby the O(n) evaluate function is not called n times.
        n = len(self.weights)
        max_value = 0
        best_mask = 0  # to store the Gray code subset
        
        total_weight = 0
        total_value = 0
        
        best_values = []
        current_values = []
        iterations_list = []
        
        iterations = min((2**n), self.algorithm_data['max_iterations_iterative'] + 1)
        # Calculate step size to reduce list to <= max_length
        step = max(1, (iterations - 1) // max_length)

        for mask in range(1, iterations):#max loops from 1 to (2**n) - 1
            gray = mask ^ (mask >> 1)  # compute Gray code i.e. 1110
            
            # Previous Gray code
            prev_gray = (mask - 1) ^ ((mask - 1) >> 1) #sub mask -1 into formula i.e. 1010
            # Bit that changed
            diff = gray ^ prev_gray #XOR because 1110 1010 gives 0100
            i = n - ((diff).bit_length())  # index of changed bit (bit length returns the number of bits to store number there 00010 returns 2)
            
            if gray & diff != 0:  # item added prev_gray = 001, gray = 011 therefore diff = 010: gray & diff = 011 & 010 = 010 which is not zero => item is added
                total_weight += self.weights[i]
                total_value += self.values[i]
            else:  # item removed
                total_weight -= self.weights[i]
                total_value -= self.values[i]
            
            if total_weight <= self.capacity and total_value > max_value:
                max_value = total_value
                best_mask = gray  # store best subset mask

            #Due to the significant amount of iterations, I decided to only record data for the graph after a fixed number of steps
            if mask % step == 0:
                best_values.append(max_value)
                current_values.append(total_value if total_weight <= self.capacity else 0)
                iterations_list.append(mask)

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        axis.plot(iterations_list, best_values, marker='o', label='Best Value')
        axis.plot(iterations_list, current_values, label='Current Value', alpha=0.5, color='grey')

        axis.set_title('Brute Force (iterative) - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Value')
        axis.legend()

        # Format mask as binary string (n bits, left to right = item 0..n-1)
        return max_value, bin(best_mask)[2:].zfill(n), fig, end_time

    def solve_01_knapsack_BF_recursive(self):
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)
        axis.axis('off')
        axis.set_title('Brute Force (recursive) - 0/1 Knapsack Problem')

        #Number of layers shown on the decision tree
        max_tree_layer = 7

        items = len(self.weights)

        #used for the decision tree when selecting the corret object to be highlighted
        path_edges = defaultdict(dict)
        #inner recursive function
        def solve(capacity=0, n=0, best_combination='', plot_pos_x=0, plot_pos_y=0, dx=20):
            if n == 0:
                capacity = self.capacity
            
            self.recursion_counter += 1
            if n == max_tree_layer and items > max_tree_layer and plot_pos_y != 0: self.plot_node(axis, plot_pos_x, plot_pos_y, lastnode=True)
            elif n <= max_tree_layer:
                #Call the plot node function which adds to the figure the marked plot at a given x, y coord
                self.plot_node(axis, plot_pos_x, plot_pos_y)

            if self.recursion_counter >= self.algorithm_data['max_iterations_recursive']:
                return (0, '')
            
            if n == len(self.weights) or capacity == 0:
                #Adds '0' to the end of the string in order to ensure it is the specified length
                return (0, best_combination.ljust(len(self.weights), '0'))
            else:
                including_item = 0
                if self.weights[n] <= capacity:
                    #including item
                    xi, yi = plot_pos_x - dx, plot_pos_y - 1
                    if n < max_tree_layer:
                        #Plots a line between previous node and new node position
                        line = axis.plot([plot_pos_x, xi], [plot_pos_y, yi], color="green", label="included", zorder=1)
                        path_edges[n][best_combination + '1'] = line
                    including_item, best_combination1 = solve(capacity-self.weights[n], n+1, best_combination + '1', xi, yi, dx / 2)
                    including_item += self.values[n]

                #excluding item
                xe, ye = plot_pos_x + dx, plot_pos_y - 1
                if n < max_tree_layer:
                    #Plots a line between previous node and new node position
                    line = axis.plot([plot_pos_x, xe], [plot_pos_y, ye], color="red", label="excluded", zorder=1)
                    path_edges[n][best_combination + '0'] = line
                excluding_item, best_combination2 = solve(capacity, n+1, best_combination + '0', xe, ye, dx/2)

                if including_item > excluding_item:
                    return (including_item, best_combination1)
                else:
                    return (excluding_item, best_combination2)
        
        value, solution = solve()

        end_time = time.perf_counter()

        #highlight selected path
        for depth in range(min(max_tree_layer, items)):
            prefix = solution[:depth + 1]
            line = path_edges[depth].get(prefix)
            if line:
                #Coloured blue and labelled select
                line[0].set_color('blue')
                line[0].set_label('selected')

        #In the table I will show the number of recursive calls
        display_text = "Number of recursive calls: " + str(self.recursion_counter) + (" (limit exceeded)" if self.recursion_counter >= self.algorithm_data['max_iterations_recursive'] else '')
        axis.text(
            0.5, 0, display_text,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="center",
        )

        #The decision tree will be incomplete if the number of recursive calls exceeds the recursion limit
        #This is because not all branches will be explored

        lines, labels = axis.get_legend_handles_labels()
        # Merge them, removing duplicates by label
        by_label = dict(zip(labels, lines))
                
        axis.legend(by_label.values(), by_label.keys(), loc='upper right')

        return value, solution, fig, end_time
                
    def solve_01_knapsack_GApprox(self, weights=[], values=[], capacity=[], get_total_weight=False, generate_graph=False):
        #To get a general approximation (this would be exact for the fractional knapsack problem), we can generate a list of value / weight
        if len(weights) == 0:
            weights = self.weights
            values = self.values
            capacity = self.capacity
        
        n = len(weights)
        items = [None] * n
        for x in range(n):
            weight = weights[x]
            value = values[x]
            items[x] = ((value/weight, weight, value, x))
        items = sorted(items, reverse=True)
        #Sorts in descending order
        total_weight = 0
        max_value = 0
        best_combination = ['0'] * n
        final_item_index = 0
        #includes item until the current item weight + total weight > capacity
        for x in range(n):
            if items[x][1] + total_weight > capacity: 
                final_item_index = x
                break
            total_weight += items[x][1]
            max_value += items[x][2]
            best_combination[items[x][3]] = '1'

        if generate_graph:
            end_time = time.perf_counter()
            fig = Figure(dpi=250)
            axis = fig.add_subplot(1, 1, 1)

            item_list = list(range(n))
            item_list_1 = item_list[:final_item_index]
            item_list_2 = item_list[final_item_index:]
            
            axis.scatter(item_list_1, [items[x][0] for x in range(final_item_index)], color='green', label='included')
            axis.scatter(item_list_2, [items[x][0] for x in range(final_item_index, n)], color='red', label='excluded')

            axis.set_title('Greedy Approximation - 0/1 Knapsack Problem')
            axis.set_xlabel('Items Sorted')
            axis.set_ylabel('Value/Weight Ratio')
            axis.legend()
            #greedy graph will have 
            return max_value, "".join(best_combination), fig, end_time
        else:
            if get_total_weight: 
                return max_value, "".join(best_combination), total_weight
            else: 
                return max_value, "".join(best_combination)

    def solve_fractional_knapsack_G(self, weights, values, capacity):
        n = len(weights)
        items = []
        #Same as in solve_01_knapsack_GApprox()
        for x in range(n):
            weight = weights[x]
            value = values[x]
            items.append((value/weight, weight, value))
        items = sorted(items, reverse=True)
        total_weight = 0
        max_value = 0
        for x in range(n):
            if items[x][1] + total_weight > capacity:
                max_value += (items[x][2] * ((capacity - total_weight)/items[x][1]))
                #Includes the maximum fractional amount of this last item
                break
            total_weight += items[x][1]
            max_value += items[x][2]
        
        return max_value
    
    def solve_01_knapsack_DP_Top_Down(self):
        memo = {}
        memo_size = []
        hits_list = []
        #follows the same recursive idea as in solve_01_knapsack_BF_recursive()
        def solve(capacity=0, n=0, best_combination=''):
            self.recursion_counter += 1
            memo_size.append(len(memo))
            hits_list.append(self.hits)
            if n == 0:
                capacity = self.capacity

            if n == len(self.weights):
                return (0, '')
            elif capacity == 0:
                return (0, '0' * (len(self.weights) - n - 1))
            elif (capacity, n) in memo:
                self.hits += 1
                return memo[(capacity, n)]
            else:
                including_item = 0
                if self.weights[n] <= capacity:
                    including_item, best_combination1 = solve(capacity-self.weights[n], n+1, best_combination)
                    including_item += self.values[n]

                excluding_item, best_combination2 = solve(capacity, n+1, best_combination)

                if including_item > excluding_item:
                    result = (including_item, '1' + best_combination1)
                else:
                    result = (excluding_item, '0' + best_combination2)

                memo[(capacity, n)] = result
                return result
        
        value, solution = solve()

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        recursions_list = list(range(self.recursion_counter))
        axis.plot(recursions_list, memo_size, label='Memo Size')
        axis.set_title('DP (Memoization) - 0/1 Knapsack Problem')
        axis.set_xlabel('Recursive calls')
        axis.set_ylabel('Count')
        axis.plot(recursions_list, hits_list, label='hits', color="green")

        axis.legend(loc='upper left')

        #Displays the number of total hits achieved
        axis.text(
            0.75, 0.10, 'Total hits: ' + str(self.hits),
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="center",
        )

        return value, solution, fig, end_time
    
    def solve_01_knapsack_DP_Bottom_Up(self):
        #Bottom up (Tabulation)
        n = len(self.weights)
        #Rows are items and columns are self.capacity
        #dp[i][c] = best value using items 0...row-1 with self.capacity x
        #therefore columns is capacity and rows is number of items 
        dp = [[0] * (self.capacity + 1) for _ in range(n+1)]
        # Fill table bottom-up
        for i in range(1, len(dp)):#i-items
            for c in range(self.capacity + 1):#c-self.capacity
                if self.weights[i-1] <= c:
                    include_item = dp[i-1][c-self.weights[i-1]] + self.values[i-1]#row above shifted along by the weight of the new item
                    exclude_item = dp[i-1][c] #row above
                    dp[i][c] = max(include_item, exclude_item)
                else:
                    dp[i][c] = dp[i-1][c] #Carry over from row above
        max_value = dp[n][self.capacity]

        #Reconstruct best_combination
        i = n
        c = self.capacity
        best_combination = ['0'] * n
        while i != 0 and c != 0:
            if dp[i][c] != dp[i-1][c]:#Different value, therefore item was picked
                best_combination[i-1] = '1'
                c -= self.weights[i-1]
                i -= 1
            else:
                i -= 1

        end_time = time.perf_counter()

        #Crops data into a labelled table to be displayed on output
        m_data = np.array(dp)
        items_list = np.array(list(range(n + 1)))
        m_data = np.column_stack((items_list, m_data))

        capacity_list = np.array([0] + list(range(self.capacity + 1)))
        m_data = np.vstack((capacity_list, m_data))
        
        data = m_data[:250, :250]

        return max_value, ''.join(best_combination), data, end_time

    def solve_01_knapsack_BnB(self):
        n = len(self.weights)
        max_value = 0
        best_solution = '0' * n
        priority_queue = []
        priority_queue_length_list = []
        branches_pruned = 0
        branches_pruned_list = []
        # Compute initial bounds
        ub_root = math.floor(self.solve_fractional_knapsack_G(self.weights, self.values, self.capacity))
        lb_root, lb_root_solution = self.solve_01_knapsack_GApprox()

        # Priority queue stores (-ub, lb, weight, value, solution, index, lb_solution)
        # Negate ub to simulate max-heap (heapq defaults to minheap). Therefore those with a maximum ub are explored first.
        # Algorithm should trap ub and lb in order to find optimal solution
        heapq.heappush(priority_queue, (-ub_root, lb_root, 0, 0, '', 0, lb_root_solution))
        iterations = 0
        while priority_queue:
            iterations += 1
            priority_queue_length_list.append(len(priority_queue))
            branches_pruned_list.append(branches_pruned)
            #pop the next node from the queue
            neg_ub, lb, current_weight, current_value, current_solution, i, lb_solution = heapq.heappop(priority_queue)
            ub = -neg_ub

            #End of branch
            if i >= n:
                continue
            
            #Prune branches
            if current_weight > self.capacity or ub <= max_value:
                branches_pruned += 1
                continue

            #set max_value
            if lb > max_value:
                max_value = lb
                best_solution = current_solution + lb_solution

            #including item
            new_weight = current_weight + self.weights[i]
            if new_weight <= self.capacity:
                new_value = current_value + self.values[i]
                remaining_capacity = self.capacity - new_weight
                #only lb is an actual solution
                ub_incl = new_value + math.floor(self.solve_fractional_knapsack_G(self.weights[i+1:], self.values[i+1:], remaining_capacity))
                lb_incl, lb_incl_solution = self.solve_01_knapsack_GApprox(self.weights[i+1:], self.values[i+1:], remaining_capacity)
                lb_incl += new_value
                #only add to the queue to be explored if the ub is greater than the max value
                if ub_incl > max_value:
                    heapq.heappush(priority_queue, (-ub_incl, lb_incl, new_weight, new_value, current_solution + '1', i + 1, lb_incl_solution))

            #excluding item
            ub_excl = current_value + math.floor(self.solve_fractional_knapsack_G(self.weights[i+1:], self.values[i+1:], self.capacity - current_weight))
            lb_excl, lb_excl_solution = self.solve_01_knapsack_GApprox(self.weights[i+1:], self.values[i+1:], self.capacity - current_weight)
            lb_excl += current_value
            if ub_excl > max_value:
                heapq.heappush(priority_queue, (-ub_excl, lb_excl, current_weight, current_value, current_solution + '0', i + 1, lb_excl_solution))

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        iterations_list = list(range(iterations))
        axis.plot(iterations_list, priority_queue_length_list, label='Priority Queue Length')
        axis.set_title('Branch and Bound - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Length')

        axis2 = axis.twinx()
        axis2.set_ylabel('Branches Pruned')
        axis2.plot(iterations_list, branches_pruned_list, label='Branches Pruned', color="green")

        # Optional: Add legends for both axes
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        return max_value, best_solution, fig, end_time

    def solve_01_knapsack_GeneticAlg(self):
        #variables
        n = len(self.weights)

        isTournamentSelection = True if self.algorithm_data['selection_type_GA'] == "Tournament selection" else False
        tournament_size = self.algorithm_data['tournament_size_GA']

        GENERATIONS = self.algorithm_data['max_iterations_GA']
        population_size = self.algorithm_data['population_size_GA']
        crossover_rate = self.algorithm_data['cross_over_rate']
        mutation_rate = self.algorithm_data['mutation_rate']
        elitism_rate = self.algorithm_data['elitism_rate']

        elitism_number = int(elitism_rate * population_size)
        global_best_combination = '0' * n
        global_max_value = 0

        population = [''] * population_size
        fitnesses = [0] * population_size
        #Graph lists
        average_fitnesses = [0] * GENERATIONS
        best_fitnesses = [0] * GENERATIONS
        lowest_fitnesses = [0] * GENERATIONS
        #Generate random population to begin
        for x in range(population_size):
            value, binary_string, weight = self.generate_starting_solution(is_greedy=False)
            population[x] = binary_string
            fitnesses[x] = value
        
        for i in range(GENERATIONS):
            new_population = [''] * population_size
            if isTournamentSelection:
                #Tournament Selection (else roulette wheel selection)
                for x in range(population_size):
                    binary_string = '0' * n
                    max_value = 0
                    for y in range(tournament_size):
                        index = random.randint(0, population_size - 1)
                        if fitnesses[index] > max_value:
                            binary_string = population[index]
                            max_value = fitnesses[index]
                    new_population[x] = binary_string
            else:
                total_fitness = sum(fitnesses)
                for x in range(population_size):
                    random_variable = random.random() * total_fitness
                    cumulative_sum = 0
                    solution_index = 0
                    for p in range(population_size):
                        cumulative_sum += fitnesses[p]
                        if random_variable <= cumulative_sum:
                            solution_index = p
                            break
                    new_population[x] = population[solution_index]

            #Cross over certain pairs of the population
            for x in range(0, population_size-1, 2):
                cross_over_random_variable = random.random()
                if cross_over_random_variable > crossover_rate: continue
                crossover_point = random.randint(0, n-1)
                
                new_population[x], new_population[x+1] = new_population[x][:crossover_point] + new_population[x+1][crossover_point:], new_population[x+1][:crossover_point] + new_population[x][crossover_point:]

            #Mutate to introduce variance
            for x in range(population_size):
                for mutation_bit in range(n):
                    mutate_random_variable = random.random()
                    if mutate_random_variable > mutation_rate: continue
                    new_population[x] = (
                        new_population[x][:mutation_bit] + 
                        ('1' if new_population[x][mutation_bit] == '0' else '0') + 
                        new_population[x][mutation_bit+1:]
                    )

            #Elitism
            indexes = heapq.nlargest(elitism_number, range(len(fitnesses)), key=fitnesses.__getitem__)
            for x in range(elitism_number):
                swapping_index = random.randint(0, population_size-1)
                new_population[swapping_index] = population[indexes[x]]

            average_fitnesses[i] = sum(fitnesses) / population_size
            best_fitnesses[i] = max(fitnesses)
            if best_fitnesses[i] > global_max_value:
                global_max_value = best_fitnesses[i]
                global_best_combination = population[fitnesses.index(global_max_value)]
            lowest_fitnesses[i] = min(x for x in fitnesses if x > 0)

            population = new_population
            fitnesses = list(map(self.evaluate, population))

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        generations = list(range(GENERATIONS))

        #Plot average, best and minimum fitness all on same graph
        axis.plot(generations, average_fitnesses, marker='o', label='Average Fitness')
        axis.plot(generations, best_fitnesses, marker='o', label='Highest Fitness')
        axis.plot(generations, lowest_fitnesses, marker='o', label='Minimum Fitness')

        axis.set_title('Genetic Algorithm - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Fitness')
        axis.legend()

        return global_max_value, global_best_combination, fig, end_time

    def solve_01_knapsack_random(self):
        n = len(self.weights)
        RUNTIME = self.algorithm_data['time_limit']
        time_interval = min(0.001, RUNTIME / 100)
        start_time = last_time = time.time()
        best_combination = '0' * n
        max_value = 0
        
        best_values = []
        current_values = []
        times_list = []
        counter = 0
        while time.time() < start_time + RUNTIME:
            #Generates a random binary string of fixed length n
            binary_string = bin(random.randint(0, (2**n)-1))[2:].zfill(n)

            value = self.evaluate(binary_string)
            if value > max_value:
                best_combination = binary_string
                max_value = value
            
            if time.time() - last_time >= time_interval:
                last_time = time.time()
                best_values.append(max_value)
                current_values.append(value)
                times_list.append(last_time - start_time)

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        axis.plot(times_list, best_values, marker='o', label='Best Value')
        axis.plot(times_list, current_values, label='Current Value', alpha=0.5, color='red')

        axis.set_title('Randomised Search - 0/1 Knapsack Problem')
        axis.set_xlabel('Time')
        axis.set_ylabel('Value')
        axis.legend()

        return max_value, best_combination, fig, end_time

    def solve_01_knapsack_SA(self):
        n = len(self.weights)
        value, solution, weight = self.generate_starting_solution(self.algorithm_data['is_greedy_SA'])
        
        max_value = value
        best_solution = solution

        initial_temperature = self.algorithm_data['initial_temperature']
        minimum_temperature = self.algorithm_data['minimum_temperature']
        alpha = self.algorithm_data['alpha']

        iteration = 0
        max_iterations = self.algorithm_data['max_iterations_SA']

        #worse = 0
        #accepted = 0
        
        range_of_data = max(self.values) - min(self.values)

        temperature = initial_temperature

        temperature_list = []
        best_values = []
        current_values = []

        while temperature >= minimum_temperature and iteration < max_iterations:
            temperature_list.append(temperature)
            Found = False
            while Found == False:
                random_bit = random.randint(0, n-1)
                if solution[random_bit] == '1':
                    new_solution = solution[:random_bit] + ('0') + solution[random_bit + 1:]
                    new_value = value - self.values[random_bit]
                    new_weight = weight - self.weights[random_bit]
                    Found = True
                else:
                    new_solution = solution[:random_bit] + ('1') + solution[random_bit + 1:]
                    new_value = value + self.values[random_bit]
                    new_weight = weight + self.weights[random_bit]

                    if new_weight <= self.capacity:
                        Found = True
            
            if new_value >= value:
                value = new_value
                weight = new_weight
                solution = new_solution

                if new_value > max_value:
                    max_value = new_value
                    best_solution = new_solution
            else:
                #Probability formula
                #Delta will normalise the range as input data may vary in terms of 'Range of values:' variable
                delta = (value - new_value) / range_of_data #relative positive 'worse' difference
                acceptance_probability = math.exp(-delta / temperature)
                #print("Difference: " + str(value-new_value) + "acceptance probability: " + str(acceptance_probability))
                rand_var = random.random()

                #worse += 1
                if rand_var < acceptance_probability:
                    value = new_value
                    weight = new_weight
                    solution = new_solution
                    #accepted += 1

            best_values.append(max_value)
            current_values.append(value)

            temperature *= alpha
            iteration += 1
        #print(accepted, worse, accepted/worse, iteration, temperature, value, max_value)
        
        # fig will include cooling schedule against iterations, current_solution and best_solution, greedy if there
        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)
        iterations_list = list(range(iteration))

        axis.set_title('Simulated Annealing - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Value')
        axis.plot(iterations_list, best_values, marker='o', label='Best Value')
        axis.plot(iterations_list, current_values, label='Current Value', alpha=0.5, color='grey')
        
        axis2 = axis.twinx()
        axis2.set_ylabel('Temperature')
        axis2.plot(iterations_list, temperature_list, label='Temperature', color="red")

        #Add legends for both axes
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        return max_value, best_solution, fig, end_time

    def solve_01_knapsack_Tabu(self):
        n = len(self.weights)
        value, solution, weight = self.generate_starting_solution(self.algorithm_data['is_greedy_tabu'])
            
        max_value = value
        best_solution = solution

        tabu_list = []
        tabu_list_maxsize = self.algorithm_data['tabu_list_size']
        ITERATIONS = self.algorithm_data['max_iterations_tabu_search']

        best_values = []
        current_values = []
        tabu_length_list = []

        for y in range(ITERATIONS):
            #Efficiently finding next move/neighbour by evaluating possible new solutions and selecting the best non-tabu solution
            best_neighbour = ''
            best_neighbour_weight = 0
            best_neighbour_value = 0
            worse = False
            for x in range(n):
                #There are n neighbours
                if solution[x] == '0':
                    #We flip to 1
                    if weight + self.weights[x] <= self.capacity:
                        new_neighbour_value = value + self.values[x]
                        new_neighbour = solution[:x] + '1' + solution[x+1:]
                        new_neighbour_weight = weight + self.weights[x]
                        if new_neighbour_value > best_neighbour_value and int(new_neighbour, 2) not in tabu_list:
                            best_neighbour_value = new_neighbour_value
                            best_neighbour = new_neighbour
                            best_neighbour_weight = new_neighbour_weight
                elif solution[x] == '1':
                    #We flip to 0
                    new_neighbour_value = value - self.values[x]
                    new_neighbour = solution[:x] + '0' + solution[x+1:]
                    new_neighbour_weight = weight - self.weights[x]
                    if new_neighbour_value > best_neighbour_value and int(new_neighbour, 2) not in tabu_list:
                        best_neighbour_value = new_neighbour_value
                        best_neighbour = new_neighbour
                        best_neighbour_weight = new_neighbour_weight
                        worse = True

            if best_neighbour_value > max_value:
                max_value = best_neighbour_value
                best_solution = best_neighbour
                if worse: print("accepted worse " + str(y))
            
            value = best_neighbour_value
            solution = best_neighbour
            weight = best_neighbour_weight

            tabu_list.append(int(solution, 2))
            
            while len(tabu_list) > tabu_list_maxsize:
                tabu_list.pop(0)
        
            best_values.append(max_value)
            current_values.append(value)
            tabu_length_list.append(len(tabu_list))

        end_time = time.perf_counter()
        #In the graph for tabu search we will include the length of the tabu list, the max value, the current value as they are the relevant properties
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)
        iterations_list = list(range(ITERATIONS))
        axis.set_title('Tabu Search - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Value')
        axis.plot(iterations_list, best_values, marker='o', label='Best Value')
        axis.plot(iterations_list, current_values, label='Current Value', alpha=0.5, color='grey')
        
        axis2 = axis.twinx()
        axis2.set_ylabel('Length')
        axis2.plot(iterations_list, tabu_length_list, label='Length of Tabu List', color="green")

        # Optional: Add legends for both axes
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        return max_value, best_solution, fig, end_time

    def solve_01_knapsack_ACO(self):
        n = len(self.weights)
        ITERATIONS = self.algorithm_data['max_iterations_ACO']
        ALPHA = self.algorithm_data['alpha_ACO']
        BETA = self.algorithm_data['beta_ACO']
        number_of_ants = self.algorithm_data['ants']
        evapouration_rate = self.algorithm_data['evapouration_rate']

        global_best_combination = ['0'] * n
        max_value = 0

        attractiveness = [self.values[i] / (self.weights[i]) for i in range(n)]

        pheremones = [0.01] * n

        best_value_list = []
        pheremones_storage_list = []

        for x in range(ITERATIONS):
            #Generate probabilities for iteration
            probabilities = [0] * n
            for i in range(n):
                probabilities[i] = (pheremones[i]**ALPHA) * (attractiveness[i]**BETA)

            #Generate solution for each ant, and update pheremones for next iteration
            for y in range(number_of_ants):
                binary_string = ['0'] * n
                current_weight = 0
                current_value = 0

                feasible_set = list(range(n))

                while feasible_set:
                    feasible_set = [i for i in feasible_set if current_weight + self.weights[i] <= self.capacity]
                    if not feasible_set: break

                    feasible_probabilities = [probabilities[i] for i in feasible_set]
                    total = sum(feasible_probabilities)

                    if total == 0: break
                    
                    Random_Variable = random.random() * total
                    cumulative_sum = 0
                    item_index = 0

                    for index, p in zip(feasible_set, feasible_probabilities):
                        cumulative_sum += p
                        if Random_Variable <= cumulative_sum:
                            item_index = index
                            break
                    
                    binary_string[item_index] = '1'
                    current_weight += self.weights[item_index]
                    current_value += self.values[item_index]
                    feasible_set.remove(item_index)

                if current_value > max_value:
                    global_best_combination = binary_string[:]
                    max_value = current_value

                try:
                    #Update pheremones
                    pheremone_strength = 1 / (1 + ((max_value - current_value) / max_value))
                except:
                    continue
                
                for w in range(n):
                    if binary_string[w] == '1':
                        pheremones[w] += pheremone_strength
            
            best_value_list.append(max_value)
            pheremones_storage_list.append(pheremones)

            #Evapouration
            pheremones = [evapouration_rate * pheremone for pheremone in pheremones]

        end_time = time.perf_counter()
        #Graph for ACO will have the following components:
        #Pheremone Strength, Attractiveness, probabilities, best value
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        iterations_list = list(range(ITERATIONS))

        axis.plot(iterations_list, best_value_list, marker='o', label='Best Value')

        axis.set_title('Ant Colony Optimisation - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Value')

        axis2 = axis.twinx()
        axis2.set_ylabel('Pheremone Strength')
        for x in range(n):
            axis2.plot(iterations_list, [pheremones_storage_list[y][x] for y in range(ITERATIONS)], alpha=0.7)

        # Create a proxy artist for the legend
        multi_coloured_proxy = Line2D([0], [0], color='black', linewidth=2)
        # Add legend manually
        lines1, labels1 = axis.get_legend_handles_labels()
        axis.legend(lines1 + [multi_coloured_proxy], labels1 + ["(Multicoloured) Pheremones"], loc="upper left")

        return max_value, "".join(global_best_combination), fig, end_time

    def solve_01_knapsack_PSO(self):
        def repair(solution, greedy_ordering):
            #ADD phase
            total_weight = sum([self.weights[x] for x in range(len(self.weights)) if solution[x] == 1])
            gap = self.capacity - total_weight
            i = 0
            new_solution = list(solution).copy()
            
            while gap > 0 and i < n:
                index = greedy_ordering[i]
                if solution[index] == 0 and gap >= self.weights[index]:
                    new_solution[index] = 1
                    gap -= self.weights[index]
                    total_weight += self.weights[index]
                i += 1
            
            #DROP phase
            over = total_weight - self.capacity
            i = 0
            while over > 0 and i < n:
                index = greedy_ordering[n-1-i]
                if solution[index] == 1:
                    new_solution[index] = 0
                    over -= self.weights[index]
                i += 1

            return new_solution

        def evaluate_PSO(integer_list):
            #uses integer lists
            total_weight = sum([self.weights[x] for x in range(len(self.weights)) if integer_list[x] == 1])
            total_value = sum([self.values[x] for x in range(len(self.values)) if integer_list[x] == 1])
            return total_value if total_weight <= self.capacity else 0
        
        #Uses list of integers until very end
        n = len(self.weights)
        max_iterations = self.algorithm_data['max_iterations_PSO']
        population = self.algorithm_data['population_size_PSO']
        inertia = self.algorithm_data['inertia']
        cognitive_factor = self.algorithm_data['cognitive_factor']
        social_factor = self.algorithm_data['social_factor']
        steepness = self.algorithm_data['steepness']
        max_velocity = self.algorithm_data['max_velocity']
        best_value_list = []

        swarm = [Particle(n) for _ in range(population)]
        iteration = 0
        gbest = []
        gbest_fitness = 0

        #Compute greedy ordering
        items = [None] * n
        for x in range(n):
            items[x] = (self.values[x]/self.weights[x], x)
        items = sorted(items, reverse=True)
        greedy_ordering = [b for (a, b) in items]
        
        while iteration < max_iterations:
            for particle in swarm:
                particle.position = repair(particle.position, greedy_ordering).copy()
                new_fitness = evaluate_PSO(particle.position)
                
                if new_fitness > particle.pbest_fitness: 
                    particle.pbest = particle.position[:]
                    particle.pbest_fitness = new_fitness

                if particle.pbest_fitness > gbest_fitness:
                    gbest = particle.pbest[:]
                    gbest_fitness = particle.pbest_fitness
                
                particle.pbest_list.append(particle.pbest_fitness)
            
            for particle in swarm:
                for x in range(n):
                    #UPDATE VELOCITY
                    particle.velocity[x] = (inertia * particle.velocity[x]) + (cognitive_factor * random.random() * ((particle.pbest[x]) - (particle.position[x]))) + (social_factor * random.random() * ((gbest[x]) - (particle.position[x])))
                    particle.velocity[x] = max(min(particle.velocity[x], max_velocity), -max_velocity)
                    #print(particle.velocity[x])
                    #UPDATE POSITION
                    if random.random() < (1 / (1 + math.exp(-steepness * particle.velocity[x]))):
                        particle.position[x] = 1
                    else:
                        particle.position[x] = 0
                
                particle.velocity_list.append(sum(particle.velocity) / len(particle.velocity))

            best_value_list.append(gbest_fitness)
            iteration += 1

        end_time = time.perf_counter()
        fig = Figure(dpi=250)
        axis = fig.add_subplot(1, 1, 1)

        iterations_list = list(range(max_iterations))

        axis.plot(iterations_list, best_value_list, marker='o', label='Best Value')

        for particle in swarm:
            axis.plot(iterations_list, particle.pbest_list, color='red', alpha=0.5, marker='s', markersize=1, label='Personal Best Value')
            
        axis.set_title('Particle Swarm Optimisation - 0/1 Knapsack Problem')
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Value')

        axis2 = axis.twinx()
        axis2.set_ylabel('Velocity')
        for particle in swarm:
            axis2.plot(iterations_list, particle.velocity_list, color='green', label='Average Velocity')

        
        # Get handles and labels from both axes
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()

        # Merge them, removing duplicates by label
        by_label = dict(zip(labels1 + labels2, lines1 + lines2))

        axis.legend(by_label.values(), by_label.keys(), loc='center right')

        return gbest_fitness, "".join(map(str, gbest)), fig, end_time

    def evaluate(self, binary_string, get_total_weight=False):
        total_value = sum([self.values[x] for x in range(len(self.values)) if binary_string[x] == "1"])
        total_weight = sum([self.weights[x] for x in range(len(self.weights)) if binary_string[x] == "1"])

        if total_weight > self.capacity:
            total_value = 0
        
        if get_total_weight:
            return total_value, total_weight
        else:
            return total_value

    def generate_starting_solution(self, is_greedy):
        if is_greedy:
            value, solution, weight = self.solve_01_knapsack_GApprox(get_total_weight=True)
        else:
            n = len(self.weights)
            solution = bin(random.randint(0, (2**n)-1))[2:].zfill(n)
            value, weight = self.evaluate(solution, get_total_weight=True)
            while value == 0:
                solution = bin(random.randint(0, (2**n)-1))[2:].zfill(n)
                value, weight = self.evaluate(solution, get_total_weight=True)
        
        return value, solution, weight

def time_and_run_binaryKP(function_index, weights, values, capacity, algorithm_data):
    problem = BinaryKP(weights, values, capacity, algorithm_data)
    functions = [problem.solve_01_knapsack_random, problem.solve_01_knapsack_BF_iterative, problem.solve_01_knapsack_BF_recursive, problem.solve_01_knapsack_GApprox, problem.solve_01_knapsack_DP_Top_Down, problem.solve_01_knapsack_DP_Bottom_Up, problem.solve_01_knapsack_BnB, problem.solve_01_knapsack_GeneticAlg, problem.solve_01_knapsack_ACO, problem.solve_01_knapsack_SA, problem.solve_01_knapsack_Tabu, problem.solve_01_knapsack_PSO]
    start_time = time.perf_counter()
    function = functions[function_index]
    if function == problem.solve_01_knapsack_GApprox:
        Answer = problem.solve_01_knapsack_GApprox(generate_graph=True)
    else:
        Answer = function()
    end_time = Answer[-1]
    elapsed_time = end_time - start_time

    fig = None
    if len(Answer) > 3:
        #There is a figure
        fig = Answer[2]
    
    solution_value, solution_weight = problem.evaluate(Answer[1], get_total_weight=True)
    return Answer[0], solution_weight, Answer[1], elapsed_time, Answer[0] == solution_value, fig

def generate_input_data_binaryKP(n, range_of_values, correlation_strength, capacity_load):
    #request.form['number_of_items'], request.form['range_of_values'], request.form['correlation_strength'], request.form['capacity_load']
    #Input data parameters
    weights = [0] * n
    values = [0] * n
    capacity = 0

    index = 0
    
    for index in range(n):
        weight = random.randint(1, range_of_values)
        weights[index] = weight

        if correlation_strength == 0:
            value = random.randint(1, range_of_values)
        else:
            value = max(1, min(range_of_values, random.randint(int(correlation_strength * weight), int((2 - correlation_strength) * weight))))
            #When correlation strength is 0 between 1 and 2W (still weakly constrained)
        values[index] = value

    capacity = max(1, min(sum(weights), int(capacity_load * sum(weights))))

    return weights, values, capacity