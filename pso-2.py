# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        pso-2
 @Created:          2024/2/3 12:21
 @Software:         PyCharm
 
 @Author:           Jiayu ZENG
 @Email:            jiayuzeng123@gmail.com
 
 @Description:      

 ------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import math

# 设置随机种子以确保结果的可重复性
random.seed(0)

# 定义障碍物：每个障碍物是一个元组，包含中心点坐标和半径
obstacles = [((2, 2), 0.5), ((4, 4), 1)]

# 定义起点和终点
start_point = (0, 0)
end_point = (5, 5)


# 定义检查点是否在障碍物内的函数
def in_obstacle(point, obstacles):
    for center, radius in obstacles:
        if np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) < radius:
            return True
    return False


# 定义适应度函数
def fitness(path, obstacles):
    # 计算路径长度
    distance = sum(np.sqrt(np.sum((np.array(path[i]) - np.array(path[i - 1])) ** 2)) for i in range(1, len(path)))
    penalty = 0
    # 对于路径中的每个点，如果它在障碍物内，则增加惩罚
    for point in path:
        if in_obstacle(point, obstacles):
            penalty += 100  # 为每个在障碍物内的点添加高额惩罚
    return distance + penalty


# 定义PSO中的粒子
class Particle:
    def __init__(self, start, end, num_points=5):
        self.num_points = num_points
        self.positions = [start] + [start + (np.random.rand(2) * (end - start)) for _ in range(num_points - 2)] + [end]
        self.velocity = [np.random.rand(2) for _ in range(num_points)]
        self.best_position = copy.deepcopy(self.positions)
        self.best_fitness = float('inf')

    def update_position(self):
        for i in range(1, self.num_points - 1):  # 不更新起点和终点
            self.positions[i] += self.velocity[i]

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=1):
        for i in range(1, self.num_points - 1):  # 不更新起点和终点的速度
            r1, r2 = np.random.rand(2)
            cognitive_velocity = c1 * r1 * (np.array(self.best_position[i]) - np.array(self.positions[i]))
            social_velocity = c2 * r2 * (np.array(global_best_position[i]) - np.array(self.positions[i]))
            self.velocity[i] = w * np.array(self.velocity[i]) + cognitive_velocity + social_velocity


# 初始化粒子群
num_particles = 50
particles = [Particle(np.array(start_point), np.array(end_point), 16) for _ in range(num_particles)]
global_best_position = copy.deepcopy(particles[0].positions)
global_best_fitness = float('inf')

# PSO主循环
iterations = 10000
for iteration in range(iterations):
    for particle in particles:
        fitness_val = fitness(particle.positions, obstacles)
        # 更新个体最优
        if fitness_val < particle.best_fitness:
            particle.best_fitness = fitness_val
            particle.best_position = copy.deepcopy(particle.positions)
        # 更新全局最优
        if fitness_val < global_best_fitness:
            global_best_fitness = fitness_val
            global_best_position = copy.deepcopy(particle.positions)

    for particle in particles:
        particle.update_velocity(global_best_position)
        particle.update_position()

    print(iteration, global_best_fitness)
    # 每10次迭代绘制一次路径
    if 0:#iteration % 10 == 0:
        plt.figure(figsize=(8, 6))
        for center, radius in obstacles:
            circle = plt.Circle(center, radius, color='r', alpha=0.5)
            plt.gca().add_patch(circle)
        xs, ys = zip(*global_best_position)
        plt.plot(xs, ys, '-o', label=f'Iteration {iteration}')
        plt.scatter(*start_point, color='green', label='Start')
        plt.scatter(*end_point, color='blue', label='End')
        plt.legend()
        plt.title(f'Iteration {iteration}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

plt.figure(figsize=(8, 6))
for center, radius in obstacles:
    circle = plt.Circle(center, radius, color='r', alpha=0.5)
    plt.gca().add_patch(circle)
xs, ys = zip(*global_best_position)
plt.plot(xs, ys, '-o', label=f'Iteration {iteration}')
plt.scatter(*start_point, color='green', label='Start')
plt.scatter(*end_point, color='blue', label='End')
plt.legend()
plt.title(f'Iteration {iteration}')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
