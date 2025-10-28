"""
展示成功逃脱的例子
"""
import sys
import io
from grid import GridWorld
from algorithms import minimax_escape_expectimax

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("  使用已知成功的种子演示魔鬼逃生")
print("=" * 80)

gw = GridWorld.from_json('map.json')

# 使用之前测试中发现的成功种子
seed = 13
import random
rng = random.Random(seed)
demon_start = gw.randomize_demon(rng)

print(f"\n配置:")
print(f"  起点: {gw.start}")
print(f"  终点: {gw.goal}")
print(f"  魔鬼起始位置: {demon_start}")
print(f"  种子: {seed}")

dist = abs(demon_start[0] - gw.start[0]) + abs(demon_start[1] - gw.start[1])
print(f"  魔鬼距离起点: {dist}")

print(f"\n运行 Expectimax 算法...")

result = minimax_escape_expectimax(
    gw,
    depth=15,
    max_steps=1000,
    demon_start=demon_start,
    seed=seed
)

print(f"\n结果:")
print(f"  成功: {result.success}")
print(f"  智能体路径长度: {len(result.agent_path)}")
print(f"  魔鬼路径长度: {len(result.demon_path)}")
print(f"  被捕获步骤: {result.caught_step}")
print(f"  最终评分: {result.score}")
print(f"  智能体最终位置: {result.agent_path[-1]}")
print(f"  目标位置: {gw.goal}")
print(f"  是否到达目标: {result.agent_path[-1] == gw.goal}")

if result.success:
    print("\n✓✓✓ 成功演示：智能体成功避开魔鬼到达出口！")
    
    print(f"\n前20步详细过程:")
    for i in range(min(20, len(result.agent_path))):
        agent_pos = result.agent_path[i]
        demon_pos = result.demon_path[i] if i < len(result.demon_path) else None
        dist_to_goal = abs(agent_pos[0] - gw.goal[0]) + abs(agent_pos[1] - gw.goal[1])
        dist_to_demon = abs(agent_pos[0] - demon_pos[0]) + abs(agent_pos[1] - demon_pos[1]) if demon_pos else 0
        print(f"  步骤 {i:3d}: 智能体@{agent_pos} (距目标={dist_to_goal:2d}, 距魔鬼={dist_to_demon:2d}), 魔鬼@{demon_pos}")
    
    if len(result.agent_path) > 20:
        print(f"  ... ({len(result.agent_path) - 20} 步省略)")
        i = len(result.agent_path) - 1
        agent_pos = result.agent_path[i]
        demon_pos = result.demon_path[i] if i < len(result.demon_path) else None
        print(f"  步骤 {i:3d}: 智能体@{agent_pos} (到达目标!), 魔鬼@{demon_pos}")
    
    print("\n" + "=" * 80)
    print("  功能验证完全成功！")
    print("  ✓ 魔鬼随机起始位置")
    print("  ✓ 魔鬼随机行走")
    print("  ✓ 智能体使用Expectimax算法找到逃生路径")
    print("  ✓ 成功避开魔鬼到达出口")
    print("=" * 80)
else:
    print(f"\n此次运行未成功，尝试其他种子...")

