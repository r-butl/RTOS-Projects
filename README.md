# Simulation of Execution Time Prediction

[![Watch the Video]([https://img.youtube.com/vi/2XYbItuuTAc/maxresdefault.jpg)](https://youtu.be/2XYbItuuTAc)

The objective of this project was to evaluate various scheduling algorithms to optimize the allocation of processes within a predefined time window. Each process is characterized by an average execution time and a variance, which serve as parameters for a normal distribution. Whenever a process is executed, an actual execution time is computed based on this distribution, utilizing the process's specified average and variance.

The simulator records historical execution times for each process, leveraging this data to predict future execution durations. It then attempts to efficiently fill the given time window with a specified number of processes. The algorithms employed by the simulator prioritize different criteria to optimize the scheduling:

- **Longest Execution Time First**: Prioritizes processes with the longest execution times.
- **Shortest Execution Time First**: Focuses on processes with the shortest execution times.
- **Lowest Variance in Execution Time First**: Selects processes with the most consistent execution times.

The results were mildly dissapointing as there was little variation between algorithms, however, it was a great excercise in creating a multithreaded simulation in python!
