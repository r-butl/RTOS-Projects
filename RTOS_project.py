#
#       RTOS Algorithm Comparison Simulator
#           by Lucas Butler
#
#       This program simulates different scheduling algorithms that use predicted execution times
#       to prioritize execution order. The objective is to maximize the number of processes executed,
#       the time utilization of the processor, and effectively predict execution time.
#
#

import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

def plot_3d(data_sets, plot_titles=[], title=""):
    ax = plt.figure().add_subplot(projection='3d')

    # Colors for different data sets
    colors = ['r', 'g', 'b']  # Red, Green, Blue

    for i, data_set in enumerate(data_sets):
        x, y, z = data_set
        ax.scatter(x, y, z, c=colors[i], label=plot_titles[i])

    ax.set_xlabel("% Utilization")
    ax.set_ylabel("Time Constraint")
    ax.set_zlabel("% Processes Missed")
    ax.set_title(title)

    # Setting the angle of the 3D plot
    ax.view_init(elev=20., azim=-35)

    # Adding a legend
    ax.legend()

    plt.show()

class process:
    
    id = 0
    priority = 0
    memory_length = 10      # Store the last n execution times, the scheduler takes the mean or variance of this array to prioritize tasks

    # Sets the sigma and mu and id, populates run time memory
    def __init__(self, id):
        self.times = [1]    # For storing the previous times
        self.sigma = np.random.choice([1, 2, 3], 1)[0]  # Unique average time for the task  
        self.mu = np.random.choice([10, 11, 12, 13], 1)[0]  # Unique variance for the task
        self.id = id    # Unique ID

        # Populate the times memory before simulation start
        for _ in range(self.memory_length):
            self.step()
            
    # Calculates a runtime, 
    def step(self):
        
        # calculate Execution time using the normal distribution and unique sigma and mu
        exe_time = round(abs(np.random.normal(self.mu, self.sigma, 1)[0]), 2)
        
        # Stored the execution time in memory
        if (len(self.times) == self.memory_length):
            del self.times[0]
        self.times.append(exe_time)

        return exe_time
    
    # Make a prediction of the next time using memory
    def predict_time(self):
        return np.mean(self.times)
    
    # Calculate the variance of the memory
    def give_variance(self):
        samples = self.times
        sum = 0
        sumsq = 0
        for x in samples:
            sum = sum + x
            sumsq = sumsq + pow(x, 2)
        mean = sum/len(samples)
        return (sumsq - len(samples)*pow(mean , 2))/(len(samples)-1)
    
    # Returns the contents of the class
    def contents(self):
        return [self.id, self.times, self.sigma, self.mu]

# Queues the tasks in order of shortest time first
def shortest_time_first(input, time_constraint):
    predicted_times = {p: p.predict_time() for p in input}  # calculate the predicted times for each task using average of memory
    return_elements = []
    total_time = 0

    # Sort by shortest time and add to the queue until no time remains 
    predicted_times = dict(sorted(predicted_times.items(), key=lambda k: k[1], reverse=True))
    for i in predicted_times.items():
        if i[1] + total_time <= time_constraint:
            return_elements.append(i)
            total_time = total_time + i[1]

    return return_elements

# Queues the tasks in order of longest time first
def longest_time_first(input, time_constraint):
    predicted_times = {p: p.predict_time() for p in input}  # calculate the predicted times for each task using average of memory
    return_elements = []
    total_time = 0

    # Sort by longest time and add to the queue until no time remains
    predicted_times = dict(sorted(predicted_times.items(), key=lambda k: k[1], reverse=False))
    for i in predicted_times.items():
        if i[1] + total_time <= time_constraint:
            return_elements.append(i)
            total_time = total_time + i[1]

    return return_elements

# Takes lowest variance first
def lowest_variance_first(input, time_constraint):

    prioritize = {p: p.give_variance()  for p in input} # Calculate the variance of each memory
    prioritize = dict(sorted(prioritize.items(), key=lambda k: k[1], reverse=False)) # Sort by lowest variance first
    predicted_times = {p: p.predict_time() for p in prioritize.keys()}
    return_elements = []
    total_time = 0

    # Add items to the queue in order of lowest variance first until no more items fit
    for i in predicted_times.items():
        if i[1] + total_time <= time_constraint:
            return_elements.append(i)
            total_time = total_time + i[1]

    return return_elements

# Simulations one algorithm and one time constraint
def simulation(algorithm, time_constraint):
        
        num_processes = 20  # Generate 20 tasks to choose randomly from
        time_steps = 1000   # Generate a large data set
        max_queue_processes = 10    # Pass 10 tasks to the scheduler
        utilization = []
        missed_execution = []
        total_processes = 1

        processes = []

        # Step up processes
        for i in range(num_processes):
            processes.append(process(i))

        for _ in range(time_steps):

            # Pick 10 tasks randomly to queue
            queued = np.random.choice(processes, max_queue_processes)
            # Apply the scheduling algorithm
            execution_order = algorithm(queued, time_constraint)    
            total_processes += len(execution_order)
            # Get actual execution times for each task
            execution_times = {p[0]: p[0].step() for p in execution_order}

            # Determine how many tasks out of the predicted execution list actual executed in the time window
            total_time = 0
            actually_executed = []
            for i in execution_order:
                if total_time + execution_times[i[0]] <= time_constraint:
                    total_time += execution_times[i[0]]
                    actually_executed.append(i[0])
                else:
                    break   # If the item did not finish in the window, we can guarantee nothing else was executed

            # Store data
            utilization.append(total_time/time_constraint)
            missed_execution.append(len(execution_order) - len(actually_executed))

        avg_util = np.mean(utilization)
        avg_missed = sum(missed_execution)/total_processes
        return avg_util, time_constraint, avg_missed

# Runs many simulations
def engine(algorithm=None):

    # Store data from simulation
    avg_util = []
    time_c = []
    avg_missed = []

    # Try time windows form 10 to 130
    time_constraint = list(range(10, 130, 1))

    partial_function = partial(simulation, algorithm)

    with Pool(processes=16) as pool:
        results = pool.map(partial_function, time_constraint)

    avg_util = [i[0] for i in results]
    time_c = [i[1] for i in results]
    avg_missed = [i[2] for i in results]
    return avg_util, time_c, avg_missed

if __name__ == "__main__":

    strategies = [longest_time_first, shortest_time_first,  lowest_variance_first]
    names = ['Longest Time First', "Shortest Time First", "Lowest Variance First"]
    results = []

    for s in strategies:
        results.append(engine(s))
 
    plot_3d(results, plot_titles=names, title="Performance of Prediction Algorithms")
