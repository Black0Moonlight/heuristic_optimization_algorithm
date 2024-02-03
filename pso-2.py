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
BOUNDARY_X = (0, 6)
BOUNDARY_Y = (0, 6)

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
    # 对路径上的每个点计算总势能
    for point in path:
        # 计算吸引势能，假设目标点是已知的
        total_potential += attractive_potential(point, end_point)

        # 计算障碍物的排斥势能
        for center, radius in obstacles:
            rep_potential = repulsive_potential(point, np.array(center), radius, influence_radius)
            if rep_potential == float('inf'):
                return float('inf')  # 如果点在障碍物内，返回无穷大的适应度值
            total_potential += rep_potential

    # 路径长度作为额外的适应度值，鼓励更短的路径
    path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))

    return total_potential + path_length


# 定义PSO中的粒子
class Particle:
    def __init__(self, start, end, num_points=10):
        self.num_points = num_points
        self.positions = [np.array(start)]
        # 计算起点到终点的向量
        line_vector = np.array(end) - np.array(start)
        # 生成一系列沿着起点到终点直线的点
        for i in range(1, num_points - 1):
            # 在起点和终点之间线性插值来找到新的点
            t = i / (num_points - 1)
            point_on_line = np.array(start) + t * line_vector
            # 添加一个小的随机偏移量
            offset = (np.random.rand(2) - 0.5) * 0.1  # 调整偏移量的大小，保证点不会偏离太远
            next_point = point_on_line + offset
            # 确保点仍然在设定的距离范围内（0.1到1之间）
            self.positions.append(self._clamp_point(next_point, self.positions[-1], 0.1, 1))
        self.positions.append(np.array(end))
        # 初始化速度为零
        self.velocity = [np.zeros(2) for _ in range(num_points)]
        self.best_position = copy.deepcopy(self.positions)
        self.best_fitness = float('inf')

    def _clamp_point(self, point, reference, min_dist, max_dist):
        """确保点与参考点之间的距离在[min_dist, max_dist]范围内"""
        direction = point - reference
        distance = np.linalg.norm(direction)
        if distance < min_dist:
            return reference + direction / distance * min_dist
        elif distance > max_dist:
            return reference + direction / distance * max_dist
        else:
            return point

    def update_position(self):
        # 更新除了起点和终点外的所有点
        for i in range(1, self.num_points - 1):
            self.positions[i] += self.velocity[i]

    def update_velocity(self, global_best_position, w=0.5, c1=1.0, c2=1.5):
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
    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
    if close:
        plt.ioff()
    else:
        plt.show()


# 初始化粒子群
num_particles = 50
num_points = 40
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
