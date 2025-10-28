"""
分析地图结构，理解为什么逃生困难
"""
import sys
import io
from grid import GridWorld
from algorithms import shortest_path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_map():
    print("=" * 70)
    print("地图分析")
    print("=" * 70)
    
    gw = GridWorld.from_json('map.json')
    
    print(f"\n基本信息:")
    print(f"  地图大小: {gw.w} x {gw.h}")
    print(f"  起点: {gw.start}")
    print(f"  终点: {gw.goal}")
    
    # 统计地形类型
    walls = sum(1 for pos, cell_type in gw.cells.items() if cell_type == '#')
    traps = sum(1 for pos, cell_type in gw.cells.items() if cell_type == 'B')
    golds = sum(1 for pos, cell_type in gw.cells.items() if cell_type == '$')
    total_cells = gw.w * gw.h
    walkable = total_cells - walls
    
    print(f"\n地形统计:")
    print(f"  总格子数: {total_cells}")
    print(f"  墙壁: {walls} ({walls/total_cells*100:.1f}%)")
    print(f"  陷阱: {traps} ({traps/total_cells*100:.1f}%)")
    print(f"  金矿: {golds} ({golds/total_cells*100:.1f}%)")
    print(f"  可行走: {walkable} ({walkable/total_cells*100:.1f}%)")
    
    # 计算从起点到终点的最短路径
    path = shortest_path(gw, gw.start, gw.goal)
    
    if path:
        print(f"\n最短路径:")
        print(f"  长度: {len(path)} 步")
        print(f"  路径: {' -> '.join([str(p) for p in path[:5]])} ... {' -> '.join([str(p) for p in path[-3:]])}")
    else:
        print(f"\n⚠ 警告: 无法从起点到达终点!")
    
    # 分析瓶颈
    print(f"\n关键观察:")
    print(f"  1. 墙壁占比{walls/total_cells*100:.1f}%，空间相对受限")
    print(f"  2. 最短路径需要{len(path) if path else 'N/A'}步")
    print(f"  3. 魔鬼随机移动在{walkable}个可行走格子中")
    print(f"  4. 在{len(path) if path else 'N/A'}步中完全避开随机移动的魔鬼概率很低")
    
    print("\n" + "=" * 70)
    print("结论:")
    print("  当前minimax实现中，魔鬼是随机移动的（而不是最优策略）")
    print("  这导致智能体难以预测并避开魔鬼的路径")
    print("  这是一个设计选择：")
    print("    - 如果魔鬼也用minimax（完全对抗），游戏会更困难")
    print("    - 当前魔鬼用随机移动，但在狭窄地图中仍然容易碰撞")
    print("\n建议方案:")
    print("  1. 增加魔鬼的可预测性（非完全随机）")
    print("  2. 使用更开阔的地图")
    print("  3. 确保魔鬼起始位置远离起点和终点")
    print("=" * 70)

if __name__ == "__main__":
    analyze_map()

