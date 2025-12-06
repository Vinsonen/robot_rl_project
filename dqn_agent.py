#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN智能体
这个文件实现了深度Q网络(DQN)算法，用于训练两轮机器人保持平衡

作者: AI助手
日期: 2025-12-06
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque

class DQNAgent:
    """
    DQN智能体类
    实现深度Q网络算法，用于决策和学习
    """
    
    def __init__(self, state_size, action_size):
        """
        初始化智能体
        
        参数:
            state_size: 状态空间大小
            action_size: 动作空间大小
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # 超参数
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95                 # 折扣因子
        self.epsilon = 1.0                # 探索率
        self.epsilon_min = 0.01           # 最小探索率
        self.epsilon_decay = 0.995        # 探索率衰减
        self.learning_rate = 0.001        # 学习率
        self.update_target_freq = 100     # 目标网络更新频率
        
        # 创建主网络和目标网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # 计数器，用于跟踪目标网络更新
        self.update_counter = 0
    
    def _build_model(self):
        """
        构建神经网络模型
        
        返回:
            编译好的神经网络模型
        """
        model = models.Sequential()
        
        # 输入层
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        
        # 隐藏层
        model.add(layers.Dense(24, activation='relu'))
        
        # 输出层
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        # 编译模型
        model.compile(
            loss='mse',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束回合
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            
        返回:
            选择的动作
        """
        # 探索: 随机选择动作
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1.0, 1.0, size=(1,))
        
        # 利用: 根据Q值选择最优动作
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return act_values[0]
    
    def replay(self, batch_size):
        """
        从回放缓冲区中采样经验进行学习
        
        参数:
            batch_size: 批量大小
        """
        if len(self.memory) < batch_size:
            return
        
        # 随机采样批量经验
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # 计算目标Q值
            target = reward
            
            if not done:
                # 如果不是结束状态，计算未来奖励的折扣和
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            # 获取当前Q值预测
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            
            # 更新目标Q值
            target_f[0] = target
            
            # 训练模型
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
    
    def load(self, name):
        """
        加载模型权重
        
        参数:
            name: 模型文件名
        """
        self.model.load_weights(name)
        self.target_model.load_weights(name)
    
    def save(self, name):
        """
        保存模型权重
        
        参数:
            name: 模型文件名
        """
        self.model.save_weights(name)


# 测试DQN智能体
if __name__ == "__main__":
    # 导入环境
    from robot_env import TwoWheelRobotEnv
    
    # 创建环境
    env = TwoWheelRobotEnv()
    
    # 获取状态和动作空间大小
    state_size = 4  # [角度, 角速度, 位置, 速度]
    action_size = 1  # 连续动作空间，但我们的DQN输出单个值
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    episodes = 10
    batch_size = 32
    
    # 训练智能体
    for e in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        done = False
        
        # 运行一个回合
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态和累积奖励
            state = next_state
            total_reward += reward
            
            # 从经验中学习
            agent.replay(batch_size)
        
        # 打印训练进度
        print(f"回合: {e+1}/{episodes}, 总奖励: {total_reward:.2f}, 探索率: {agent.epsilon:.4f}")
    
    # 保存模型
    agent.save("dqn_robot_model.h5")
    
    # 关闭环境
    env.close()