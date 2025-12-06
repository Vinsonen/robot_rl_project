#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练智能体
这个文件用于训练两轮机器人的DQN智能体

作者: AI助手
日期: 2025-12-06
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from robot_env import TwoWheelRobotEnv
from dqn_agent import DQNAgent

def train_agent(episodes=1000, batch_size=32, render_every=50):
    """
    训练智能体
    
    参数:
        episodes: 训练回合数
        batch_size: 批量大小
        render_every: 每隔多少回合渲染一次
    
    返回:
        训练历史
    """
    # 创建环境
    env = TwoWheelRobotEnv(render_mode=None)
    
    # 获取状态和动作空间大小
    state_size = 4  # [角度, 角速度, 位置, 速度]
    action_size = 1  # 连续动作空间
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 训练历史
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'epsilon': []
    }
    
    # 开始训练
    print("开始训练智能体...")
    start_time = time.time()
    
    for e in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # 设置渲染模式
        if (e + 1) % render_every == 0:
            env.render_mode = 'human'
        else:
            env.render_mode = None
        
        # 运行一个回合
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态和累积奖励
            state = next_state
            total_reward += reward
            steps += 1
            
            # 从经验中学习
            agent.replay(batch_size)
        
        # 记录训练历史
        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(steps)
        history['epsilon'].append(agent.epsilon)
        
        # 打印训练进度
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            avg_length = np.mean(history['episode_lengths'][-10:])
            elapsed_time = time.time() - start_time
            
            print(f"回合: {e+1}/{episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"平均长度: {avg_length:.1f}, "
                  f"探索率: {agent.epsilon:.4f}, "
                  f"时间: {elapsed_time:.1f}s")
        
        # 每100回合保存一次模型
        if (e + 1) % 100 == 0:
            model_name = f"dqn_robot_model_ep{e+1}.h5"
            agent.save(model_name)
            print(f"模型已保存: {model_name}")
    
    # 保存最终模型
    agent.save("dqn_robot_model_final.h5")
    print(f"最终模型已保存: dqn_robot_model_final.h5")
    
    # 关闭环境
    env.close()
    
    # 训练总时间
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.1f}s")
    
    return history

def plot_training_history(history):
    """
    绘制训练历史
    
    参数:
        history: 训练历史字典
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    ax1.plot(history['episode_rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # 绘制回合长度曲线
    ax2.plot(history['episode_lengths'])
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.grid(True)
    
    # 绘制探索率曲线
    ax3.plot(history['epsilon'])
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # 训练参数
    episodes = 500
    batch_size = 32
    render_every = 100
    
    # 训练智能体
    history = train_agent(episodes, batch_size, render_every)
    
    # 绘制训练历史
    plot_training_history(history)