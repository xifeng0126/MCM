#遗传算法，计算机模拟生物进化过程，通过选择、交叉、变异等操作，产生新的解，逐步逼近最优解。
from sko.GA import GA_TSP
import numpy as np
import matplotlib.pyplot as plt
#解决TSP问题，即解决从起点出发，经过所有点后回到起点的最短路径问题

def cal_total_distance(points):
    num_points, = points.shape
    total_distance = 0
    for i in range(num_points - 1):
        total_distance += distance_matrix[points[i], points[i + 1]]
    total_distance += distance_matrix[points[i + 1], points[0]]
    return total_distance

num_points = 8

points = range(num_points)
points_coordinate = np.random.rand(num_points, 2)
distance_matrix = np.zeros(shape=(num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
print('distance_matrix is: \n', distance_matrix)

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=200, prob_mut=0.001)
best_points, best_distance = ga_tsp.fit()

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()