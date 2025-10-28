#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证所有算法是否正常工作"""

from grid import GridWorld
from algorithms import (
    value_iteration_plan, 
    policy_iteration_plan, 
    q_learning_plan,
    dfs_search_steps,
    greedy_search_steps,
    astar_search_steps,
    best_collect_all_golds_dp
)

def main():
    # 加载地图
    gw = GridWorld.from_json()
    
    print("=" * 60)
    print("GridWorld 算法验证")
    print("=" * 60)
    print(f"起点: {gw.start}")
    print(f"终点: {gw.goal}")
    print(f"地图大小: {gw.w} x {gw.h}")
    print()
    
    # 测试搜索算法
    print("1. DFS搜索...")
    dfs_steps = list(dfs_search_steps(gw, gw.start, gw.goal))
    print(f"   ✓ 完成，生成了 {len(dfs_steps)} 个步骤")
    
    print("2. 贪心搜索...")
    greedy_steps = list(greedy_search_steps(gw, gw.start, gw.goal))
    print(f"   ✓ 完成，生成了 {len(greedy_steps)} 个步骤")
    
    print("3. A*搜索...")
    astar_steps = list(astar_search_steps(gw, gw.start, gw.goal))
    print(f"   ✓ 完成，生成了 {len(astar_steps)} 个步骤")
    
    # 测试收集所有金币
    print("4. 收集所有金币（DP）...")
    collect_result = best_collect_all_golds_dp(gw)
    print(f"   ✓ 完成，路径长度: {len(collect_result.path)}, 总奖励: {collect_result.total_reward}")
    
    # 测试值迭代
    print("5. 值迭代...")
    value_result = value_iteration_plan(gw, gamma=0.9, theta=1e-4, max_iterations=1000)
    reached_goal_vi = value_result.path[-1] == gw.goal if value_result.path else False
    print(f"   ✓ 完成，路径长度: {len(value_result.path)}")
    print(f"   收敛: {value_result.converged}, 迭代次数: {value_result.iterations}")
    print(f"   到达终点: {'✓ 是' if reached_goal_vi else '✗ 否'}")
    
    # 测试策略迭代
    print("6. 策略迭代...")
    policy_result = policy_iteration_plan(gw, gamma=0.9, theta=1e-4, max_policy_iterations=100)
    reached_goal_pi = policy_result.path[-1] == gw.goal if policy_result.path else False
    print(f"   ✓ 完成，路径长度: {len(policy_result.path)}")
    print(f"   策略迭代次数: {policy_result.policy_iterations}")
    print(f"   到达终点: {'✓ 是' if reached_goal_pi else '✗ 否'}")
    
    # 测试Q学习
    print("7. Q学习...")
    qlearning_result = q_learning_plan(gw, episodes=500, gamma=0.9, alpha=0.1, epsilon=0.1, seed=42)
    reached_goal_ql = qlearning_result.path[-1] == gw.goal if qlearning_result.path else False
    avg_reward = sum(qlearning_result.episode_rewards[-100:]) / min(100, len(qlearning_result.episode_rewards)) if qlearning_result.episode_rewards else 0
    print(f"   ✓ 完成，路径长度: {len(qlearning_result.path)}")
    print(f"   训练回合: {qlearning_result.episodes}")
    print(f"   平均奖励（最后100回合）: {avg_reward:.2f}")
    print(f"   到达终点: {'✓ 是' if reached_goal_ql else '✗ 否'}")
    
    # 总结
    print()
    print("=" * 60)
    print("验证结果总结")
    print("=" * 60)
    all_reached = reached_goal_vi and reached_goal_pi and reached_goal_ql
    if all_reached:
        print("✓ 所有算法都成功到达终点！")
    else:
        print("✗ 部分算法未能到达终点：")
        if not reached_goal_vi:
            print("  - 值迭代")
        if not reached_goal_pi:
            print("  - 策略迭代")
        if not reached_goal_ql:
            print("  - Q学习")
    
    return all_reached

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

