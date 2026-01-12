# moaco_distance_fare.py

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("delhi_metro_updated2.0 (1).csv")
data = data.head(300)  # keep same size as MOPSO

distance = data["Distance_km"].values
fare = data["Fare"].values
n_dim = len(distance)

# ===============================
# 2. ACO Parameters
# ===============================
n_ants = 40
n_iters = 100
rho = 0.2          # evaporation
alpha = 1.0        # pheromone importance
beta = 2.0         # heuristic importance
tau0 = 0.1         # initial pheromone

# ===============================
# 3. Initialize Pheromones
# ===============================
pheromone_dist = np.ones(n_dim) * tau0
pheromone_fare = np.ones(n_dim) * tau0

heuristic_dist = 1 / (distance + 1e-10)
heuristic_fare = 1 / (fare + 1e-10)

archive = []  # Pareto archive

# ===============================
# 4. Objective Function
# ===============================
def objectives(solution):
    total_dist = np.sum(solution * distance)
    total_fare = np.sum(solution * fare)
    return np.array([total_dist, total_fare])

# ===============================
# 5. Dominance Check
# ===============================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# ===============================
# 6. MOACO Main Loop
# ===============================
start_time = time.time()

for it in range(n_iters):
    solutions = []

    for ant in range(n_ants):
        solution = np.zeros(n_dim, dtype=int)

        for j in range(n_dim):
            tau_d = pheromone_dist[j] ** alpha
            tau_f = pheromone_fare[j] ** alpha
            eta_d = heuristic_dist[j] ** beta
            eta_f = heuristic_fare[j] ** beta

            prob = (tau_d * eta_d + tau_f * eta_f)
            prob = prob / (1 + prob)

            solution[j] = np.random.rand() < prob

        obj = objectives(solution)
        solutions.append((obj, solution))

    # Evaporation
    pheromone_dist *= (1 - rho)
    pheromone_fare *= (1 - rho)

    # Pareto Update + Pheromone Reinforcement
    for obj, sol in solutions:
        dominated = False
        new_archive = []

        for a in archive:
            if dominates(a[0], obj):
                dominated = True
                break
            if not dominates(obj, a[0]):
                new_archive.append(a)

        if not dominated:
            new_archive.append((obj, sol))
            archive = new_archive

            # Reinforce pheromone
            pheromone_dist += sol / (obj[0] + 1e-10)
            pheromone_fare += sol / (obj[1] + 1e-10)

end_time = time.time()

# ===============================
# 7. Extract Pareto Front
# ===============================
pareto_dist = [a[0][0] for a in archive]
pareto_fare = [a[0][1] for a in archive]

print("===================================")
print("MOACO COMPLETED")
print("Iterations:", n_iters)
print("Ants:", n_ants)
print("Pareto Solutions:", len(archive))
print("Execution Time:", round(end_time - start_time, 3), "seconds")
print("===================================")

# ===============================
# 8. Plot Pareto Front
# ===============================
plt.figure(figsize=(8, 5))
plt.scatter(pareto_dist, pareto_fare)
plt.xlabel("Total Distance (Minimize)")
plt.ylabel("Total Fare (Minimize)")
plt.title("Pareto Front — MOACO (Distance vs Fare)")
plt.grid(True)
plt.show()
