#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人强化学习环境
这个文件定义了一个简单的两轮机器人环境，用于强化学习训练

作者: AI助手
日期: 2025-12-06
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class TwoWheelRobotEnv:
    """
    两轮机器人环境类
    模拟一个两轮自平衡机器人的物理特性和环境交互
    """
    
    def __init__(self, render_mode=None):
        """
        初始化环境
        
        参数:
            render_mode: 渲染模式，None表示不渲染，'human'表示实时渲染
        """
        # 物理参数
        self.gravity = 9.8  # 重力加速度 (m/s^2)
        self.mass = 1.0     # 机器人质量 (kg)
        self.length = 0.5   # 机器人高度 (m)
        self.dt = 0.05      # 时间步长 (s)
        self.max_steps = 500  # 每回合最大步数
        
        # 状态空间: [角度, 角速度, 位置, 速度]
        self.observation_space = {
            'low': np.array([-np.pi/2, -10.0, -10.0, -5.0]),
            'high': np.array([np.pi/2, 10.0, 10.0, 5.0])
        }
        
        # 动作空间: [-1.0, 1.0] 表示施加的力
        self.action_space = {
            'low': np.array([-1.0]),
            'high': np.array([1.0])
        }
        
        # 渲染模式
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.robot_line = None
        self.trail_line = None
        self.trail = []
        
        # 重置环境
        self.reset()
    
    def reset(self):
        """
        重置环境到初始状态
        
        返回:
            初始状态
        """
        # 初始状态: 小角度偏差，其他为0
        self.angle = np.random.uniform(-0.1, 0.1)  # 初始角度
        self.angular_velocity = 0.0                # 初始角速度
        self.position = 0.0                        # 初始位置
        self.velocity = 0.0                        # 初始速度
        self.step_count = 0                        # 步数计数器
        
        # 重置轨迹
        self.trail = [(self.position, 0)]
        
        # 返回初始观测
        return self._get_observation()
    
    def _get_observation(self):
        """
        获取当前观测状态
        
        返回:
            当前状态的numpy数组
        """
        return np.array([self.angle, self.angular_velocity, self.position, self.velocity])
    
    def step(self, action):
        """
        执行一步环境交互
        
        参数:
            action: 智能体选择的动作
            
        返回:
            observation: 新的观测状态
            reward: 获得的奖励
            done: 是否结束回合
            info: 额外信息
        """
        # 确保动作在有效范围内
        action = np.clip(action, self.action_space['low'], self.action_space['high'])[0]
        
        # 物理更新 (简化的动力学模型)
        # 角度加速度 = 重力项 + 控制项
        angle_acceleration = (self.gravity / self.length) * np.sin(self.angle) + action
        
        # 更新角速度和角度
        self.angular_velocity += angle_acceleration * self.dt
        self.angle += self.angular_velocity * self.dt
        
        # 更新速度和位置
        self.velocity += action * self.dt
        self.position += self.velocity * self.dt
        
        # 更新步数
        self.step_count += 1
        
        # 更新轨迹
        self.trail.append((self.position, 0))
        if len(self.trail) > 50:  # 限制轨迹长度
            self.trail.pop(0)
        
        # 计算奖励
        # 1. 保持平衡的奖励 (角度越小奖励越高)
        balance_reward = 1.0 - 5.0 * abs(self.angle)
        
        # 2. 保持在中心位置的奖励
        position_reward = 1.0 - 0.1 * abs(self.position)
        
        # 3. 保持低速的奖励
        velocity_reward = 1.0 - 0.1 * abs(self.velocity)
        
        # 总奖励
        reward = balance_reward + position_reward + velocity_reward
        
        # 检查是否结束回合
        done = False
        
        # 角度过大导致摔倒
        if abs(self.angle) > np.pi/4:
            done = True
            reward -= 10.0  # 摔倒惩罚
        
        # 位置超出范围
        if abs(self.position) > 5.0:
            done = True
            reward -= 5.0   # 超出范围惩罚
        
        # 达到最大步数
        if self.step_count >= self.max_steps:
            done = True
        
        # 获取当前观测
        observation = self._get_observation()
        
        # 额外信息
        info = {
            'angle': self.angle,
            'position': self.position,
            'steps': self.step_count
        }
        
        # 渲染 (如果需要)
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, done, info
    
    def render(self):
        """
        渲染环境
        使用matplotlib绘制机器人状态
        """
        if self.fig is None:
            # 创建图形和坐标轴
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_xlim(-6, 6)
            self.ax.set_ylim(-1, 2)
            self.ax.set_title('Two-Wheel Robot Simulation')
            self.ax.set_xlabel('Position (m)')
            self.ax.set_ylabel('Height (m)')
            self.ax.grid(True)
            
            # 绘制地面
            self.ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
            
            # 初始化机器人图形
            robot_x = [self.position - 0.1, self.position + 0.1]
            robot_y = [0, 0]
            self.robot_line, = self.ax.plot(robot_x, robot_y, 'ro-', linewidth=3)
            
            # 初始化轨迹
            self.trail_line, = self.ax.plot([], [], 'b--', alpha=0.5)
        
        # 更新机器人位置
        robot_height = self.length * np.cos(self.angle)
        robot_top_x = self.position + self.length * np.sin(self.angle)
        robot_top_y = robot_height
        
        # 更新机器人图形
        self.robot_line.set_data(
            [self.position - 0.1, self.position + 0.1, robot_top_x, self.position - 0.1],
            [0, 0, robot_top_y, 0]
        )
        
        # 更新轨迹
        if len(self.trail) > 1:
            trail_x, trail_y = zip(*self.trail)
            self.trail_line.set_data(trail_x, trail_y)
        
        # 更新图形
        plt.pause(0.01)
    
    def close(self):
        """
        关闭环境
        清理资源
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.robot_line = None
            self.trail_line = None


# 测试环境
if __name__ == "__main__":
    # 创建环境
    env = TwoWheelRobotEnv(render_mode='human')
    
    # 重置环境
    observation = env.reset()
    print(f"初始观测: {observation}")
    
    # 运行几个随机动作
    total_reward = 0
    done = False
    
    while not done:
        # 随机选择动作
        action = np.random.uniform(-1.0, 1.0, size=(1,))
        
        # 执行动作
        observation, reward, done, info = env.step(action)
        
        # 累积奖励
        total_reward += reward
        
        # 打印信息
        print(f"步骤: {info['steps']}, 角度: {info['angle']:.3f}, 位置: {info['position']:.3f}, 奖励: {reward:.3f}")
        
        # 小延迟，便于观察
        time.sleep(0.05)
    
    print(f"总奖励: {total_reward}")
    
    # 关闭环境
    env.close()