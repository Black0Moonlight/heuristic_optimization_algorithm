# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        pso-2
 @Created:          2024/2/3 12:21
 @Software:         PyCharm
 
 @Author:           Jiayu ZENG
 @Email:            jiayuzeng123@gmail.com
 
 @Description:      运行在自己写的环境

 ------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import math

# 设置随机种子以确保结果的可重复性
random.seed(0)
plt.ion()

# 定义障碍物：每个障碍物是一个元组，包含中心点坐标和半径
obstacles = [((2, 2), 0.5), ((3, 4), 0.5), ((1, 3), 0.5)]

# 定义起点和终点
start_point = (0, 0)
end_point = (5, 5)
ful_dis = np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)


# 定义检查点是否在障碍物内的函数
def in_obstacle(point, obstacles):
    dis = 0
    for center, radius in obstacles:
        dis = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
        if dis < radius:
            return 1, (radius - dis)/radius
        else:
            return 0, (dis - radius)/dis


# 地图的边界值
BOUNDARY_X = (-1, 6)
BOUNDARY_Y = (-1, 6)

# 定义适应度函数
def attractive_potential(current_point, goal_point, scale=1):
    # 使用简单的欧几里得距离作为吸引势能
    return scale * np.linalg.norm(current_point - goal_point)


def repulsive_potential(current_point, obstacle_point, obstacle_radius, influence_radius, scale=1):
    # 计算障碍物的排斥势能
    d = np.linalg.norm(current_point - obstacle_point)
    if d <= obstacle_radius:
        # 如果在障碍物内部，则给予很大的排斥势能
        return float('inf')
    elif obstacle_radius < d <= influence_radius:
        # 如果在影响范围内，则根据距离计算排斥势能
        return scale * (1 / d - 1 / influence_radius) ** 2
    else:
        # 如果在影响范围外，则没有排斥势能
        return 0


def fitness(path, obstacles, influence_radius=1):
    total_potential = 0
    angle_penalty = 0
    distance_penalties = 0
    for point in path:
        # Attracted potential energy
        total_potential += attractive_potential(point, end_point)

        # Repelling potential energy
        for center, radius in obstacles:
            rep_potential = repulsive_potential(point, np.array(center), radius, influence_radius)
            if rep_potential == float('inf'):
                return float('inf')
            total_potential += rep_potential

    # length
    path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))

    # Calculate distances between consecutive points and their variance as penalty
    distances = [np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path))]
    mean_distance = np.mean(distances)
    distance_var = np.mean([(d - mean_distance) ** 2 for d in distances])
    distance_penalties = distance_var * 0

    # angle penalty
    for i in range(1, len(path) - 1):
        vector1 = np.array(path[i]) - np.array(path[i - 1])
        vector2 = np.array(path[i + 1]) - np.array(path[i])
        if np.linalg.norm(vector1) > 0 and np.linalg.norm(vector2) > 0:
            cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if not np.isclose(angle, 0):
                angle_penalty += (1 - cos_angle) * 1

    return total_potential + path_length + angle_penalty + distance_penalties


# 定义PSO中的粒子
class Particle:
    def __init__(self, start, end, num_points=10):
        self.num_points = num_points
        self.positions = [np.array(start)]
        while len(self.positions) < num_points - 1:
            next_point = self.positions[-1] + (np.random.rand(2) - 0.5)
            if len(self.positions) > 1:  # 至少有两个点时才检查角度
                vector1 = self.positions[-2] - self.positions[-1]
                vector2 = next_point - self.positions[-1]
                if np.linalg.norm(vector1) > 0 and np.linalg.norm(vector2) > 0:
                    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                    # 如果角度大于90度，重新生成点
                    if cos_angle < 0:  # cos(90°) = 0，cos(角度)小于0意味着角度大于90°
                        continue  # 重新生成这个点
            self.positions.append(next_point)
        self.positions.append(np.array(end))
        # 初始化速度为零
        self.velocity = [np.zeros(2) for _ in range(num_points)]
        self.best_position = copy.deepcopy(self.positions)
        self.best_fitness = float('inf')

    def update_position(self):
        # 更新除了起点和终点外的所有点
        for i in range(1, self.num_points - 1):
            self.positions[i] += self.velocity[i]

    def update_velocity(self, global_best_position, w=0.8, c1=0.35, c2=1.5):
        for i in range(1, self.num_points - 1):  # 不更新起点和终点的速度
            r1, r2 = np.random.rand(2)
            cognitive_velocity = c1 * r1 * (np.array(self.best_position[i]) - np.array(self.positions[i]))
            social_velocity = c2 * r2 * (np.array(global_best_position[i]) - np.array(self.positions[i]))
            self.velocity[i] = w * np.array(self.velocity[i]) + cognitive_velocity + social_velocity


def plot_path(obstacles, path, start, end, iteration, close):
    # plt.figure(figsize=(6, 6))
    plt.clf()
    for center, radius in obstacles:
        circle = plt.Circle(center, radius, color='r', alpha=0.5)
        plt.gca().add_patch(circle)
    plt.scatter(start[0], start[1], color='red', label='Start')  # 显示起点
    plt.scatter(end[0], end[1], color='blue', label='End')  # 显示终点
    xs, ys = zip(*path)
    plt.plot(xs, ys, '-*', label=f'Iteration {iteration}')
    plt.legend()
    plt.title(f'Iteration {iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.pause(0.001)
    if close:
        plt.ioff()
    else:
        plt.show()


# 初始化粒子群
num_particles = 50
num_points = 20
particles = [Particle(np.array(start_point), np.array(end_point), num_points) for _ in range(num_particles)]
global_best_position = copy.deepcopy(particles[0].positions)
global_best_fitness = float('inf')


plot_path(obstacles, global_best_position, start_point, end_point, 0, 1)

# PSO主循环
iterations = 1000
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
    plot_path(obstacles, global_best_position, start_point, end_point, iteration, 1)
    # 每10次迭代绘制一次路径
    if iteration % 200 == 0:
        #plot_path(obstacles, global_best_position, start_point, end_point, iteration, 1)
        print(iteration, global_best_fitness)
        print("######")

plot_path(obstacles, global_best_position, start_point, end_point, iterations, 0)
