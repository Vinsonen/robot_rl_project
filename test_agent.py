#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试智能体
这个文件用于测试训练好的DQN智能体

作者: AI助手
日期: 2025-12-06
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from robot_env import TwoWheelRobotEnv
from dqn_agent import DQNAgent

def test_agent(model_path, episodes=5):
    """
    测试训练好的智能体
    
    参数:
        model_path: 模型文件路径
        episodes: 测试回合数
    """
    # 创建环境
    env = TwoWheelRobotEnv(render_mode='human')
    
    # 获取状态和动作空间大小
    state_size = 4  # [角度, 角速度, 位置, 速度]
    action_size = 1  # 连续动作空间
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 加载模型
    try:
        agent.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设置为评估模式（关闭探索）
    agent.epsilon = 0.0
    
    # 测试结果
    test_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'max_angles': [],
        'max_positions': []
    }
    
    # 开始测试
    print("开始测试智能体...")
    
    for e in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_angle = 0
        max_position = 0
        
        # 运行一个回合
        while not done:
            # 选择动作（无探索）
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新状态和累积奖励
            state = next_state
            total_reward += reward
            steps += 1
            
            # 记录最大角度和位置
            max_angle = max(max_angle, abs(info['angle']))
            max_position = max(max_position, abs(info['position']))
            
            # 小延迟，便于观察
            time.sleep(0.05)
        
        # 记录测试结果
        test_results['episode_rewards'].append(total_reward)
        test_results['episode_lengths'].append(steps)
        test_results['max_angles'].append(max_angle)
        test_results['max_positions'].append(max_position)
        
        # 打印测试结果
        print(f"测试回合: {e+1}/{episodes}, "
              f"奖励: {total_reward:.2f}, "
              f"步数: {steps}, "
              f"最大角度: {max_angle:.3f}, "
              f"最大位置: {max_position:.3f}")
    
    # 计算平均结果
    avg_reward = np.mean(test_results['episode_rewards'])
    avg_length = np.mean(test_results['episode_lengths'])
    avg_max_angle = np.mean(test_results['max_angles'])
    avg_max_position = np.mean(test_results['max_positions'])
    
    print("\n测试总结:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    print(f"平均最大角度: {avg_max_angle:.3f}")
    print(f"平均最大位置: {avg_max_position:.3f}")
    
    # 关闭环境
    env.close()
    
    return test_results

def create_animation(model_path, episode_length=200):
    """
    创建智能体运行的动画
    
    参数:
        model_path: 模型文件路径
        episode_length: 动画长度（步数）
    """
    # 创建环境
    env = TwoWheelRobotEnv(render_mode=None)
    
    # 获取状态和动作空间大小
    state_size = 4
    action_size = 1
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 加载模型
    agent.load(model_path)
    agent.epsilon = 0.0  # 关闭探索
    
    # 重置环境
    state = env.reset()
    
    # 创建动画数据
    positions = []
    angles = []
    actions = []
    
    # 运行几个步骤
    for _ in range(episode_length):
        # 选择动作
        action = agent.act(state)
        
        # 执行动作
        next_state, _, done, info = env.step(action)
        
        # 记录数据
        positions.append(info['position'])
        angles.append(info['angle'])
        actions.append(action[0])
        
        # 更新状态
        state = next_state
        
        if done:
            break
    
    # 创建动画
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 设置坐标轴
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-1, 2)
    ax1.set_title('Robot Simulation')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    ax1.grid(True)
    
    ax2.set_xlim(0, episode_length)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title('Action History')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Action')
    ax2.grid(True)
    
    # 绘制地面
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    # 初始化机器人图形
    robot_line, = ax1.plot([], [], 'ro-', linewidth=3)
    trail_line, = ax1.plot([], [], 'b--', alpha=0.5)
    
    # 初始化动作曲线
    action_line, = ax2.plot([], [], 'g-', linewidth=2)
    
    # 轨迹
    trail = []
    
    def update(frame):
        """更新动画帧"""
        # 获取当前数据
        pos = positions[frame]
        angle = angles[frame]
        
        # 更新轨迹
        trail.append((pos, 0))
        if len(trail) > 50:
            trail.pop(0)
        
        # 计算机器人位置
        robot_height = env.length * np.cos(angle)
        robot_top_x = pos + env.length * np.sin(angle)
        robot_top_y = robot_height
        
        # 更新机器人图形
        robot_line.set_data(
            [pos - 0.1, pos + 0.1, robot_top_x, pos - 0.1],
            [0, 0, robot_top_y, 0]
        )
        
        # 更新轨迹
        if len(trail) > 1:
            trail_x, trail_y = zip(*trail)
            trail_line.set_data(trail_x, trail_y)
        
        # 更新动作曲线
        action_line.set_data(range(frame+1), actions[:frame+1])
        
        return robot_line, trail_line, action_line
    
    # 创建动画
    ani = FuncAnimation(
        fig, update, frames=len(positions),
        interval=50, blit=True
    )
    
    # 保存动画
    ani.save('robot_simulation.gif', writer='pillow', fps=20)
    print("动画已保存为: robot_simulation.gif")
    
    # 显示动画
    plt.show()
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 测试参数
    model_path = "dqn_robot_model_final.h5"
    test_episodes = 5
    
    # 测试智能体
    test_results = test_agent(model_path, test_episodes)
    
    # 创建动画
    create_animation(model_path)