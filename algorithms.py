from __future__ import annotations
from typing import Tuple, Dict, List, Optional, Set, Mapping, DefaultDict
import heapq, itertools, random
from collections import defaultdict

from dataclasses import dataclass, field

from grid import GridWorld, Coord


# ---- Global action definitions (used by MDP/RL and transition function) ----
ACTIONS = {
    'R': (1, 0),
    'L': (-1, 0),
    'D': (0, 1),
    'U': (0, -1),
}
# Tie-breaking order for planning/RL; also used when enumerating actions
ACTION_ORDER = ['R', 'D', 'L', 'U']


# ------------------ Utilities ------------------
def manhattan(a, b):
    """
    计算两个坐标点之间的曼哈顿距离（网格中水平/垂直移动的步数总和）。

    曼哈顿距离公式：|x1 - x2| + |y1 - y2|，常用于路径规划中的启发式函数。
    参数:
        a - 第一个点坐标 (x, y)
        b - 第二个点坐标 (x, y)
    返回值:
        int - 两点间的曼哈顿距离
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 新增：路径重构（按父指针回溯）
def reconstruct(parent, cur):
    """
    根据 A* 等搜索中维护的 parent 字典，从 cur 回溯得到完整路径。
    parent[start] 应为 None。
    """
    path = []
    t = cur
    while t is not None and t in parent:
        path.append(t)
        t = parent[t]
    return list(reversed(path)) if path else []

# 新增：基于 BFS 的最短步数距离（缓存到指定目标）
_goal_steps_cache: Dict[Tuple[int, Coord], Dict[Coord, int]] = {}

def goal_steps_map(gw, goal):
    """
    返回从任意格到 goal 的最短步数距离表（BFS），带缓存。
    """
    key = (id(gw), goal)
    if key in _goal_steps_cache:
        return _goal_steps_cache[key]
    from collections import deque
    dist: Dict[Coord, int] = {goal: 0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        d = dist[cur]
        for nb in gw.neighbors4(cur):
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    _goal_steps_cache[key] = dist
    return dist

def grid_steps_dist(gw, a, b):
    """
    返回 a 到 b 的最短步数距离（不可达时返回一个大值）。
    """
    if b is None:
        return gw.w + gw.h
    dist_map = goal_steps_map(gw, b)
    return dist_map.get(a, gw.w + gw.h)


# ------------------ Part 1: Step-by-step search generators ------------------
@dataclass
class SearchStep:
    """搜索过程快照数据类，用于记录搜索算法的中间状态（可视化/调试用）"""
    algo: str  # 算法名称（如"DFS"、"A*"）
    current: Optional[Coord]  # 当前处理的节点坐标（None表示初始状态）
    frontier: List[Coord]  # 前沿队列（待探索节点列表）
    visited: Set[Coord]  # 已访问节点集合
    parent: Dict[Coord, Optional[Coord]]  # 当前父节点映射关系


def dfs_search_steps(gw, start, goal):
    """
    深度优先搜索(DFS)的分步执行生成器，每次迭代返回搜索过程快照。

    DFS使用栈实现，优先探索深度方向，适用于迷宫类问题，但不保证最短路径。
    生成器会逐步产出SearchStep对象，可用于动画演示搜索过程。
    参数:
        gw: GridWorld - 网格环境对象
        start - 起点坐标
        goal - 目标点坐标
    生成值:
        SearchStep - 包含当前搜索状态的快照对象
    """
    stack = [start]  # DFS使用栈存储待探索节点（LIFO）
    parent: Dict[Coord, Optional[Coord]] = {start: None}  # 父节点记录
    seen: Set[Coord] = set()  # 已访问节点集合（避免重复探索）

    # 初始状态快照（未开始搜索）
    yield SearchStep("DFS", None, list(stack), set(), dict(parent))

    while stack:
        cur = stack.pop()  # 弹出栈顶节点（深度优先）

        if cur in seen:  # 跳过已访问节点
            continue
        seen.add(cur)  # 标记为已访问

        # 产出当前节点处理状态快照
        yield SearchStep("DFS", cur, list(stack), set(seen), dict(parent))

        if cur == goal:  # 找到目标，终止搜索
            break

        # 探索当前节点的四邻域（按ACTION_ORDER顺序的逆序入栈，保证探索顺序）
        for nb in gw.neighbors4(cur):
            if nb not in parent:  # 未被发现的节点
                parent[nb] = cur  # 记录父节点
                stack.append(nb)  # 入栈待探索
                # 产出添加新节点后的快照
                yield SearchStep("DFS", cur, list(stack), set(seen), dict(parent))

    # 搜索结束快照（包含最终路径信息）
    yield SearchStep("DFS", goal if goal in parent else cur, list(stack), set(seen), dict(parent))


def greedy_search_steps(gw, start, goal):
    """
    贪婪最佳优先搜索的分步执行生成器，每次迭代返回搜索过程快照。

    贪婪搜索使用优先队列（按到目标的曼哈顿距离排序），仅关注当前最优选择，
    可能陷入局部最优，但搜索速度快。生成器用于演示搜索过程。
    参数:
        gw: GridWorld - 网格环境对象
        start - 起点坐标
        goal - 目标点坐标
    生成值:
        SearchStep - 包含当前搜索状态的快照对象
    """
    # 优先队列：(优先级, 节点)，优先级改为到目标的最短步数距离（BFS）
    pq: List[Tuple[int, Coord]] = []
    heapq.heappush(pq, (grid_steps_dist(gw, start, goal), start))
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    seen: Set[Coord] = set()

    # 初始状态快照
    yield SearchStep("Greedy", None, [start], set(), dict(parent))

    while pq:
        _, cur = heapq.heappop(pq)  # 弹出优先级最高的节点

        if cur in seen:  # 跳过已访问节点
            continue
        seen.add(cur)  # 标记为已访问

        # 提取当前前沿队列（仅保留节点坐标）
        frontier_list = [c for _, c in pq]
        yield SearchStep("Greedy", cur, frontier_list, set(seen), dict(parent))

        if cur == goal:  # 找到目标，终止搜索
            break

        # 探索四邻域节点
        for nb in gw.neighbors4(cur):
            if nb not in parent:  # 未被发现的节点
                parent[nb] = cur
                # 按到目标的距离设置优先级并入队
                heapq.heappush(pq, (grid_steps_dist(gw, nb, goal), nb))
                # 更新前沿队列并产出快照
                frontier_list = [c for _, c in pq]
                yield SearchStep("Greedy", cur, frontier_list, set(seen), dict(parent))

    # 搜索结束快照
    yield SearchStep("Greedy", goal if goal in parent else cur, [c for _, c in pq], set(seen), dict(parent))


def astar_search_steps(gw, start, goal):
    """
    A*搜索的分步执行生成器，每次迭代返回搜索过程快照。

    A*算法结合Dijkstra算法（考虑实际代价）和贪婪搜索（考虑启发式代价），
    通过f = g + h（g:起点到当前代价，h:当前到目标启发式距离）进行优先级排序，
    保证找到最短路径。生成器用于演示搜索过程。
    参数:
        gw: GridWorld - 网格环境对象
        start - 起点坐标
        goal - 目标点坐标
    生成值:
        SearchStep - 包含当前搜索状态的快照对象
    """
    openpq: List[Tuple[int, Coord]] = []  # A*优先队列：(f值, 节点)
    gcost: Dict[Coord, int] = {start: 0}  # 起点到各节点的实际代价（g值）
    parent: Dict[Coord, Optional[Coord]] = {start: None}  # 父节点记录
    goal_map = goal_steps_map(gw, goal)
    heapq.heappush(openpq, (goal_map.get(start, gw.w + gw.h), start))  # 初始节点入队
    seen: Set[Coord] = set()  # 已访问节点集合

    # 初始状态快照
    yield SearchStep("A*", None, [start], set(), dict(parent))

    while openpq:
        _, cur = heapq.heappop(openpq)  # 弹出f值最小的节点

        if cur in seen:
            continue
        seen.add(cur)

        # 产出当前节点处理状态快照
        frontier_list = [c for _, c in openpq]
        yield SearchStep("A*", cur, frontier_list, set(seen), dict(parent))

        if cur == goal:  # 找到目标，终止搜索
            break

        # 探索四邻域节点
        for nb in gw.neighbors4(cur):
            ng = gcost[cur] + 1  # 计算新代价（每步代价为1）
            # 若节点未访问或发现更优路径（更小g值）
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng  # 更新实际代价
                parent[nb] = cur  # 更新父节点
                f = ng + goal_map.get(nb, gw.w + gw.h)  # 计算f值（g+h）
                heapq.heappush(openpq, (f, nb))  # 入队

                # 更新前沿队列并产出快照
                frontier_list = [c for _, c in openpq]
                yield SearchStep("A*", cur, frontier_list, set(seen), dict(parent))

    # 搜索结束快照
    yield SearchStep("A*", goal if goal in parent else cur, [c for _, c in openpq], set(seen), dict(parent))


def a_star(gw, start, goal):
    """
    A*路径搜索算法实现，寻找从起点到目标点的最短路径。

    A*是一种启发式搜索算法，通过综合考虑从起点到当前节点的实际代价（g）和
    当前节点到目标的估计代价（h，使用曼哈顿距离），保证找到最短路径。
    参数:
        gw: GridWorld - 网格环境对象
        start - 起点坐标
        goal - 目标点坐标
    返回值:
        Optional[List[Coord]] - 最短路径坐标列表（起点→目标），无解时返回None
    """
    openpq: List[Tuple[int, Coord]] = []  # 优先队列：(f值, 节点)
    gcost: Dict[Coord, int] = {start: 0}  # 实际代价字典（g值）
    parent: Dict[Coord, Optional[Coord]] = {start: None}  # 父节点映射
    goal_map = goal_steps_map(gw, goal)
    heapq.heappush(openpq, (goal_map.get(start, gw.w + gw.h), start))

    while openpq:
        _, cur = heapq.heappop(openpq)  # 选择f值最小的节点

        if cur == goal:  # 到达目标，重构路径并返回
            return reconstruct(parent, cur)

        # 探索所有四邻域节点
        for nb in gw.neighbors4(cur):
            ng = gcost[cur] + 1  # 计算新路径代价（步长为1）
            # 若节点未访问或发现更优路径（更小g值）
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng  # 更新g值
                parent[nb] = cur  # 记录父节点
                f = ng + goal_map.get(nb, gw.w + gw.h)  # 计算f值
                heapq.heappush(openpq, (f, nb))  # 入队

    return None  # 目标不可达，返回None


def shortest_path(gw, s, t):
    """
    最短路径查询函数（A*算法封装）。

    作为A*算法的便捷调用接口，统一网格环境中的最短路径查询入口。
    参数:
        gw: GridWorld - 网格环境对象
        s - 起点坐标
        t - 目标点坐标
    返回值:
        Optional[List[Coord]] - 最短路径列表，无解时返回None
    """
    return a_star(gw, s, t)


# ------------------ Part 2: Rewards + Collect-All-Golds ------------------
@dataclass
class TrajectoryResult:
    """轨迹结果数据类，用于存储路径和对应的总奖励"""
    path: List[Coord]  # 坐标路径列表（起点→终点）
    total_reward: int  # 该路径的总奖励值


def total_reward_for_path(gw, path):
    """
    计算给定路径在网格环境中的总奖励值（基于预设奖励规则）。

    奖励规则：
    - 每移动一步：-1分（步数惩罚，鼓励短路径）
    - 踏入陷阱('B')：额外-10分（危险惩罚）
    - 首次收集金矿('$')：+10分（资源奖励，仅首次有效）
    - 到达终点('G')：+20分（目标奖励）
    参数:
        gw: GridWorld - 网格环境对象（提供单元格类型查询）
        path: List[Coord] - 待评估的路径坐标列表
    返回值:
        int - 计算得到的总奖励值（路径为空时返回-1e9惩罚）
    """
    if not path:  # 空路径返回极大惩罚
        return -10 ** 9

    reward = 0
    taken: Set[Coord] = set()  # 已收集金矿位置记录（避免重复奖励）

    # 遍历路径中的每一步移动（从path[i]到path[i+1]）
    for i in range(len(path) - 1):
        cur, nxt = path[i], path[i + 1]
        reward -= 1  # 基础步数惩罚

        # 根据目标单元格类型计算奖励
        t = gw.cell_type(nxt)
        if t == 'B':  # 陷阱惩罚
            reward += -10
        if t == '$' and nxt not in taken:  # 首次收集金矿奖励
            reward += 10
            taken.add(nxt)

    # 到达终点额外奖励
    if path[-1] == gw.goal:
        reward += 20

    return reward


def best_collect_all_golds_then_exit(gw, max_perm = 7):
    """
    寻找收集所有金矿后到达终点的最优路径（基于奖励最大化）。

    策略：
    1. 当金矿数量≤max_perm时：枚举所有金矿收集顺序（全排列），选择总奖励最高的路径
    2. 当金矿数量>max_perm时：使用贪心启发式（优先收集最近的金矿）降低计算复杂度
    参数:
        gw: GridWorld - 网格环境对象
        max_perm: int - 全排列策略的最大金矿数量阈值（默认7，超过则启用贪心）
    返回值:
        TrajectoryResult - 包含最优路径和总奖励的结果对象
    """
    golds = [p for p, t in gw.cells.items() if t == '$']  # 所有金矿位置
    s, g = gw.start, gw.goal  # 起点和终点

    # 无金矿时直接返回起点到终点的最短路径
    if not golds:
        p = shortest_path(gw, s, g) or []
        return TrajectoryResult(p, total_reward_for_path(gw, p))

    # 预处理所有关键点间的最短路径（起点、所有金矿、终点）
    points = [s] + golds + [g]
    sp_cache: Dict[Tuple[Coord, Coord], List[Coord]] = {}  # 路径缓存（避免重复计算）

    # 预计算所有点对间的最短路径
    for a, b in itertools.permutations(points, 2):
        if (a, b) not in sp_cache:
            sp_cache[(a, b)] = shortest_path(gw, a, b) or []

    best_path: List[Coord] = []
    best_score = -10 ** 9  # 初始最佳奖励（极小值）

    # 策略1：金矿数量较少时，使用全排列枚举所有可能的收集顺序
    if len(golds) <= max_perm:
        # 遍历所有金矿收集顺序的排列
        for perm in itertools.permutations(golds):
            seq = [s] + list(perm) + [g]  # 完整路径序列：起点→金矿1→...→金矿n→终点
            path: List[Coord] = [seq[0]]  # 初始化路径（起点）
            ok = True  # 路径有效性标志

            # 依次拼接序列中相邻点的最短路径
            for i in range(len(seq) - 1):
                seg = sp_cache.get((seq[i], seq[i + 1])) or []
                if not seg:  # 某段路径不存在，标记为无效
                    ok = False
                    break
                path.extend(seg[1:])  # 拼接路径（跳过起点避免重复）

            if not ok:
                continue  # 跳过无效路径

            # 计算当前排列的总奖励
            score = total_reward_for_path(gw, path)
            # 更新最佳路径
            if score > best_score:
                best_score, best_path = score, path

    # 策略2：金矿数量较多时，使用贪心启发式（优先收集最近的金矿）
    else:
        remaining = set(golds)  # 剩余未收集金矿
        cur = s  # 当前位置（起点）
        path: List[Coord] = [cur]  # 初始化路径

        # 逐个收集剩余金矿
        while remaining:
            # 使用最短步数距离选择最近的金矿
            target = min(remaining, key=lambda z: grid_steps_dist(gw, cur, z))
            # 获取当前位置到目标金矿的路径
            seg = sp_cache.get((cur, target)) or shortest_path(gw, cur, target) or []

            if not seg:  # 路径不存在，终止收集
                break

            path.extend(seg[1:])  # 拼接路径（跳过起点）
            cur = target  # 更新当前位置
            remaining.remove(target)  # 标记金矿为已收集

        # 收集完所有金矿后，前往终点
        seg = sp_cache.get((cur, g)) or shortest_path(gw, cur, g) or []
        path.extend(seg[1:])

        best_path = path
        best_score = total_reward_for_path(gw, best_path)

    return TrajectoryResult(best_path, best_score)

# ------------------ Demon escape via minimax ------------------

@dataclass
class MinimaxEscapeResult:
    """极小极大算法逃生模拟的结果数据类，存储逃生过程的关键信息"""
    agent_path: List[Coord]  # 智能体的移动路径坐标列表（起点→当前位置）
    demon_path: List[Coord]  # 恶魔的移动路径坐标列表（起点→当前位置）
    success: bool  # 逃生是否成功（智能体到达目标且未被捕获）
    demon_start: Optional[Coord]  # 恶魔的初始位置（None表示随机生成）
    caught_step: Optional[int]  # 智能体被捕获的步数（None表示未被捕获）
    score: int  # 最终评分（基于极小极大算法的评估结果）


def minimax_escape_expectimax(
        gw,
        depth = 10,
        max_steps = 1000,
        demon_start = None,
        seed = None,
):
    """
    使用Expectimax算法模拟智能体从随机移动魔鬼中逃生的过程。
    
    与minimax不同，Expectimax假设魔鬼是随机移动的（而非最优对抗），
    因此在魔鬼回合计算所有可能移动的期望值，而非最坏情况。
    这更符合魔鬼随机移动的实际行为。
    
    参数:
        gw: GridWorld - 网格世界环境对象
        depth: int - 搜索深度
        max_steps: int - 最大模拟步数
        demon_start: Optional[Coord] - 恶魔的初始位置
        seed: Optional[int] - 随机数种子
        
    返回值:
        MinimaxEscapeResult - 逃生结果
    """
    depth = max(1, depth)
    rng = random.Random(seed)
    start_agent = gw.start
    start_goal = gw.goal
    
    demon_pos = demon_start if demon_start is not None else gw.demon
    if demon_pos is None:
        demon_pos = gw.randomize_demon(rng)
    
    agent_pos = start_agent
    agent_path: List[Coord] = [agent_pos]
    demon_path: List[Coord] = [demon_pos] if demon_pos is not None else []
    opening_score: Optional[float] = None
    caught_step: Optional[int] = None
    
    WIN_SCORE = 1000
    LOSE_SCORE = -1000
    
    # 预计算到目标的距离表（BFS）
    from collections import deque
    def build_goal_dist():
        """使用BFS预计算所有位置到目标的最短距离"""
        INF = 10**6
        dist = {}
        q = deque([start_goal])
        dist[start_goal] = 0
        
        while q:
            cur = q.popleft()
            d = dist[cur]
            for nb in gw.neighbors4(cur):
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        return dist
    
    goal_dist = build_goal_dist()
    
    # 使用缓存优化魔鬼距离计算
    demon_dist_cache: Dict[Coord, Dict[Coord, int]] = {}
    
    def get_demon_dist(demon_pos):
        """获取从魔鬼位置到所有位置的距离（带缓存）"""
        if demon_pos in demon_dist_cache:
            return demon_dist_cache[demon_pos]
        
        INF = 10**6
        dist = {}
        q = deque([demon_pos])
        dist[demon_pos] = 0
        
        while q:
            cur = q.popleft()
            d = dist[cur]
            for nb in gw.neighbors4(cur):
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        
        demon_dist_cache[demon_pos] = dist
        return dist
    
    def heuristic(a, d, step):
        """改进的启发式函数：使用BFS距离 + 步数惩罚"""
        if d is None:
            dist_demon = gw.w + gw.h
        else:
            demon_dists = get_demon_dist(d)
            dist_demon = demon_dists.get(a, gw.w + gw.h)
        
        dist_goal = goal_dist.get(a, gw.w + gw.h)
        
        # 启发式：远离魔鬼 + 靠近目标 - 步数惩罚
        return int(2.0 * dist_demon - 3.0 * dist_goal - 0.1 * step)

    def expectimax(agent, demon, remaining, maximizing, memo):
        """
        Expectimax算法：智能体最大化，魔鬼取期望值（所有移动的平均）
        """
        key = (agent, demon, remaining, maximizing)
        if key in memo:
            return memo[key]
        
        current_step = depth - remaining + len(agent_path)
        
        # 终止条件
        if agent == demon:
            memo[key] = float(LOSE_SCORE + 0.1 * current_step)
            return memo[key]
        
        if agent == start_goal:
            memo[key] = float(WIN_SCORE - 0.1 * current_step)
            return memo[key]
        
        if remaining == 0:
            memo[key] = heuristic(agent, demon, current_step)
            return memo[key]
        
        # 智能体回合：最大化
        if maximizing:
            value = float('-inf')
            moves = gw.neighbors4(agent)
            if not moves:
                moves = [agent]
            
            for nxt in moves:
                if nxt == demon:
                    cand = float(LOSE_SCORE + remaining - 1)
                else:
                    cand = expectimax(nxt, demon, remaining - 1, False, memo)
                value = max(value, cand)
            
            memo[key] = value
            return value
        
        # 魔鬼回合：取期望值（随机移动）
        else:
            moves = gw.neighbors4(demon)
            if not moves:
                moves = [demon]
            
            # 计算所有可能移动的平均值
            total = 0.0
            for nxt in moves:
                if nxt == agent:
                    cand = float(LOSE_SCORE + remaining - 1)
                else:
                    cand = expectimax(agent, nxt, remaining - 1, True, memo)
                total += cand
            
            value = total / len(moves)
            memo[key] = value
            return value

    def choose_best(agent, demon):
        memo = {}
        best_val = float('-inf')
        best_move = None
        moves = gw.neighbors4(agent)
        if not moves:
            moves = [agent]
        for nxt in moves:
            if nxt == demon:
                cand = float(LOSE_SCORE + depth - 1)
            else:
                cand = expectimax(nxt, demon, depth - 1, False, memo)
            prefer = True if best_move is None else goal_dist.get(nxt, gw.w + gw.h) < goal_dist.get(best_move, gw.w + gw.h)
            if cand > best_val or (cand == best_val and prefer):
                best_val = cand
                best_move = nxt
        return best_val, best_move

    # 主模拟循环
    for step in range(max_steps):
        if agent_pos == start_goal:
            break
        if demon_pos is not None and agent_pos == demon_pos:
            caught_step = step
            break
        
        if demon_pos is None:
            demon_path = []
        
        score, move = choose_best(agent_pos, demon_pos) if demon_pos is not None else (float(WIN_SCORE), None)
        if opening_score is None:
            opening_score = score
        
        if move is None:
            break
        
        # 记录移动前的位置
        agent_prev = agent_path[-1] if len(agent_path) >= 1 else agent_pos
        demon_prev = demon_path[-1] if demon_path else demon_pos
        
        agent_pos = move
        agent_path.append(agent_pos)
        
        if demon_pos is not None:
            if agent_pos == demon_pos:
                caught_step = step + 1
                break
            
            demon_moves = gw.neighbors4(demon_pos)
            if not demon_moves:
                demon_moves = [demon_pos]
            demon_next = rng.choice(demon_moves)
            
            # 检测交叉移动（对穿）：智能体和魔鬼互换位置
            # 如果智能体从A移到B，魔鬼从B移到A，则取消魔鬼的移动
            if agent_prev == demon_next and demon_prev == agent_pos:
                # 检测到对穿，魔鬼原地不动
                demon_pos = demon_prev
            else:
                demon_pos = demon_next
            
            demon_path.append(demon_pos)
            
            if agent_pos == demon_pos:
                caught_step = step + 1
                break
        
        if agent_pos == start_goal:
            break
    
    success = agent_path[-1] == start_goal and (demon_pos is None or agent_path[-1] != demon_pos)
    final_score = int(opening_score if opening_score is not None else heuristic(agent_path[-1], demon_pos))
    return MinimaxEscapeResult(agent_path, demon_path, success, demon_start or gw.demon, caught_step, final_score)


def minimax_escape(
        gw,
        depth = 10,
        max_steps = 1000,
        demon_start = None,
        seed = None,
):
    """
    使用极小极大（Minimax）算法模拟智能体从恶魔追捕中逃生的过程。

    智能体通过极小极大算法（带alpha-beta剪枝）预测未来状态，选择最优逃生路径；
    恶魔采用随机移动策略。模拟持续到智能体到达目标、被捕获或达到最大步数。

    参数:
        gw: GridWorld - 网格世界环境对象，提供地图布局和状态查询接口
        depth: int - 极小极大算法的搜索深度（默认为6，越大决策越精准但计算量越大）
        max_steps: int - 最大模拟步数（防止无限循环，默认为80）
        demon_start: Optional[Coord] - 恶魔的初始位置（None时随机生成）
        seed: Optional[int] - 随机数种子（用于恶魔移动的可复现性，默认为None）

    返回值:
        MinimaxEscapeResult - 包含逃生路径、结果状态和评分的详细结果对象
    """
    # 确保搜索深度至少为1（避免无效搜索）
    depth = max(1, depth)
    # 初始化随机数生成器（用于恶魔的随机移动）
    rng = random.Random(seed)
    start_agent = gw.start  # 智能体起点（网格世界的起点位置）
    start_goal = gw.goal  # 目标点（智能体需要到达的逃生位置）

    # 初始化恶魔位置：优先使用指定位置，否则使用网格世界默认恶魔位置，均无则随机生成
    demon_pos = demon_start if demon_start is not None else gw.demon
    if demon_pos is None:
        demon_pos = gw.randomize_demon(rng)

    # 初始化智能体和恶魔的路径跟踪
    agent_pos = start_agent  # 智能体当前位置（初始为起点）
    agent_path: List[Coord] = [agent_pos]  # 智能体路径列表（记录移动历史）
    demon_path: List[Coord] = [demon_pos] if demon_pos is not None else []  # 恶魔路径列表
    opening_score: Optional[int] = None  # 初始评分（首次决策的评估分数）
    caught_step: Optional[int] = None  # 被捕获的步数（初始为None表示未被捕获）

    # 评分常量定义
    WIN_SCORE = 1000  # 到达目标的基础奖励分数
    LOSE_SCORE = -1000  # 被恶魔捕获的基础惩罚分数

    # 预计算到目标的距离表（BFS）
    from collections import deque
    def build_goal_dist():
        """使用BFS预计算所有位置到目标的最短距离"""
        dist = {}
        q = deque([start_goal])
        dist[start_goal] = 0
        
        while q:
            cur = q.popleft()
            d = dist[cur]
            for nb in gw.neighbors4(cur):
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        return dist
    
    goal_dist = build_goal_dist()
    
    # 使用缓存优化魔鬼距离计算
    demon_dist_cache: Dict[Coord, Dict[Coord, int]] = {}
    
    def get_demon_dist(demon_pos):
        """获取从魔鬼位置到所有位置的距离（带缓存）"""
        if demon_pos in demon_dist_cache:
            return demon_dist_cache[demon_pos]
        
        INF = 10**6
        dist = {}
        q = deque([demon_pos])
        dist[demon_pos] = 0
        
        while q:
            cur = q.popleft()
            d = dist[cur]
            for nb in gw.neighbors4(cur):
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        
        demon_dist_cache[demon_pos] = dist
        return dist

    def heuristic(a, d, step):
        """改进的启发式函数：使用BFS距离 + 步数惩罚"""
        if d is None:
            dist_demon = gw.w + gw.h
        else:
            demon_dists = get_demon_dist(d)
            dist_demon = demon_dists.get(a, gw.w + gw.h)
        
        dist_goal = goal_dist.get(a, gw.w + gw.h)
        
        # 启发式：远离魔鬼 + 靠近目标 - 步数惩罚
        return int(2.0 * dist_demon - 3.0 * dist_goal - 0.1 * step)

    def minimax(agent, demon, remaining, maximizing, alpha, beta, memo):
        """
        极小极大算法实现（带alpha-beta剪枝）：模拟智能体与恶魔的对抗决策过程。

        智能体（最大化玩家）试图最大化评分（逃离恶魔、到达目标），
        恶魔（最小化玩家）试图最小化评分（捕获智能体）。通过递归搜索未来状态，
        结合alpha-beta剪枝减少无效搜索分支，提升效率。

        参数:
            agent - 智能体当前位置
            demon - 恶魔当前位置
            remaining: int - 剩余搜索深度（递归终止条件）
            maximizing: bool - 是否为最大化玩家回合（True：智能体，False：恶魔）
            alpha: float - alpha值（最大化玩家的当前最优值，用于剪枝）
            beta: float - beta值（最小化玩家的当前最优值，用于剪枝）
            memo: Dict[...] - 记忆化字典（缓存已计算状态，避免重复计算）
        返回值:
            int - 当前状态的评估分数
        """
        # 构建状态键（智能体位置、恶魔位置、剩余深度、当前玩家）
        key = (agent, demon, remaining, maximizing)
        # 若状态已缓存，直接返回缓存结果（记忆化搜索）
        if key in memo:
            return memo[key]

        current_step = depth - remaining + len(agent_path)
        
        # 基础情况1：智能体被恶魔捕获 → 返回失败分数
        if agent == demon:
            val = LOSE_SCORE + int(0.1 * current_step)
            memo[key] = int(val)
            return memo[key]

        # 基础情况2：智能体到达目标 → 返回胜利分数
        if agent == start_goal:
            val = WIN_SCORE - int(0.1 * current_step)
            memo[key] = int(val)
            return memo[key]

        # 基础情况3：搜索深度耗尽 → 使用启发函数评估当前状态
        if remaining == 0:
            val = heuristic(agent, demon, current_step)
            memo[key] = int(val)
            return memo[key]

        # 最大化玩家回合（智能体决策）：选择所有可能移动中的最优解
        if maximizing:
            value = float('-inf')  # 初始值设为负无穷（寻找最大值）
            # 获取智能体的所有四方向邻居（可行移动）
            moves = gw.neighbors4(agent)
            if not moves:  # 若无可行移动（被墙壁包围），只能原地不动
                moves = [agent]

            # 评估每个可能的移动
            for nxt in moves:
                if nxt == demon:  # 移动后立即被恶魔捕获
                    cand = LOSE_SCORE + remaining - 1  # 剩余深度减1（下一步）
                else:  # 递归评估恶魔回合的状态
                    cand = minimax(nxt, demon, remaining - 1, False, alpha, beta, memo)

                # 更新当前最大值
                value = max(value, cand)
                # 更新alpha值（最大化玩家的最优下界）
                alpha = max(alpha, value)
                if alpha >= beta:  # alpha-beta剪枝：最小化玩家不会选择此分支
                    break

            memo[key] = int(value)
            return memo[key]

        # 最小化玩家回合（恶魔决策）：选择所有可能移动中的最差解（对智能体而言）
        else:
            value = float('inf')  # 初始值设为正无穷（寻找最小值）
            # 获取恶魔的所有四方向邻居（可行移动）
            moves = gw.neighbors4(demon)
            if not moves:  # 若无可行移动，恶魔原地不动
                moves = [demon]

            # 评估每个可能的移动
            for nxt in moves:
                if nxt == agent:  # 移动后立即捕获智能体
                    cand = LOSE_SCORE + remaining - 1
                else:  # 递归评估智能体回合的状态
                    cand = minimax(agent, nxt, remaining - 1, True, alpha, beta, memo)

                # 更新当前最小值
                value = min(value, cand)
                # 更新beta值（最小化玩家的最优上界）
                beta = min(beta, value)
                if alpha >= beta:  # alpha-beta剪枝：最大化玩家不会选择此分支
                    break

            memo[key] = int(value)
            return memo[key]

    def choose_best(agent, demon):
        memo = {}
        best_val = float('-inf')
        best_move = None
        moves = gw.neighbors4(agent)
        if not moves:
            moves = [agent]
        for nxt in moves:
            if nxt == demon:
                cand = LOSE_SCORE + depth - 1
            else:
                cand = minimax(nxt, demon, depth - 1, False, float('-inf'), float('inf'), memo)
            # 使用 BFS 步数距离的同分偏好
            prefer = True if best_move is None else goal_dist.get(nxt, gw.w + gw.h) < goal_dist.get(best_move, gw.w + gw.h)
            if cand > best_val or (cand == best_val and prefer):
                best_val = cand
                best_move = nxt
        return best_val, best_move

    # 主模拟循环
    for step in range(max_steps):
        if agent_pos == start_goal:
            break
        if demon_pos is not None and agent_pos == demon_pos:
            caught_step = step
            break

        if demon_pos is None:
            demon_path = []

        score, move = choose_best(agent_pos, demon_pos) if demon_pos is not None else (float(WIN_SCORE), None)
        if opening_score is None:
            opening_score = score

        if move is None:
            break

        # 记录移动前的位置
        agent_prev = agent_path[-1] if len(agent_path) >= 1 else agent_pos
        demon_prev = demon_path[-1] if demon_path else demon_pos
        
        agent_pos = move
        agent_path.append(agent_pos)

        if demon_pos is not None:
            if agent_pos == demon_pos:
                caught_step = step + 1
                break

            demon_moves = gw.neighbors4(demon_pos)
            if not demon_moves:
                demon_moves = [demon_pos]
            demon_next = rng.choice(demon_moves)
            
            # 检测交叉移动（对穿）：智能体和魔鬼互换位置
            # 如果智能体从A移到B，魔鬼从B移到A，则取消魔鬼的移动
            if agent_prev == demon_next and demon_prev == agent_pos:
                # 检测到对穿，魔鬼原地不动
                demon_pos = demon_prev
            else:
                demon_pos = demon_next
            
            demon_path.append(demon_pos)

            if agent_pos == demon_pos:
                caught_step = step + 1
                break

        if agent_pos == start_goal:
            break

    success = agent_path[-1] == start_goal and (demon_pos is None or agent_path[-1] != demon_pos)
    final_score = opening_score if opening_score is not None else heuristic(agent_path[-1], demon_pos, len(agent_path))
    return MinimaxEscapeResult(agent_path, demon_path, success, demon_start or gw.demon, caught_step, final_score)

# --------- Movement-cost A* with optional gold avoidance (for DP) ---------
def movement_cost(gw, from_p, to_p):
    """
    计算网格中从一个位置移动到另一个位置的代价（考虑地形类型）。

    基础移动代价为1，不同地形会附加额外代价，用于A*算法的路径成本评估。
    参数:
        gw: GridWorld - 网格世界环境对象（提供单元格类型查询）
        from_p - 起始位置坐标
        to_p - 目标位置坐标（主要根据此位置类型计算代价）
    返回值:
        int - 总移动代价（基础代价+地形附加代价）
    """
    # 基础移动代价：每步1分
    c = 1
    # 获取目标位置的单元格类型（决定附加代价）
    tp = gw.cell_type(to_p)
    if tp == 'B':  # 陷阱（'B'）：附加10分代价（危险区域惩罚）
        c += 10
    if tp == '#':  # 特殊地形（'#'）：附加5分代价（难走区域惩罚）
        c += 5
    if tp == 'P':  # 普通障碍（'P'）：附加1分代价（轻微阻碍惩罚）
        c += 1
    return c


def a_star_movement(gw, start, goal, avoid = None):
    """
    带移动代价和避障功能的A*路径搜索算法。

    扩展标准A*算法，支持指定需要避开的坐标集合（如未收集的金矿），并使用movement_cost计算路径代价，
    适用于需要考虑地形代价的场景（如动态规划中的子问题路径计算）。

    参数:
        gw: GridWorld - 网格环境对象
        start - 起点坐标
        goal - 目标点坐标
        avoid: Optional[Set[Coord]] - 需要避开的坐标集合（目标点除外），默认为空集
    返回值:
        Tuple[Optional[List[Coord]], Optional[int]] -
            若找到路径：(路径坐标列表, 路径总代价)；若未找到：(None, None)
    """
    # 初始化避障集合（默认空集）
    avoid = avoid or set()
    # 优先队列（open list）：存储待探索节点，按f值（总代价=实际代价+启发式距离）排序
    openpq: List[Tuple[int, Coord]] = []
    # gcost字典：记录从起点到每个节点的实际代价（使用movement_cost计算）
    gcost: Dict[Coord, int] = {start: 0}
    # parent字典：记录每个节点的父节点，用于路径重构
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    goal_map = goal_steps_map(gw, goal)
    heapq.heappush(openpq, (goal_map.get(start, gw.w + gw.h), start))

    # 主搜索循环：当优先队列非空时持续探索
    while openpq:
        # 弹出f值最小的节点（当前最优路径节点）
        _, cur = heapq.heappop(openpq)

        # 若到达目标节点，重构路径并返回（路径+总代价）
        if cur == goal:
            return reconstruct(parent, cur), gcost[cur]

        # 遍历当前节点的四方向邻居（上/下/左/右）
        for nb in gw.neighbors4(cur):
            # 跳过避障节点（目标节点除外，即使在避障集合中也允许通过）
            if nb in avoid and nb != goal:
                continue

            # 计算从起点→当前节点→邻居的新实际代价（当前代价+移动到邻居的代价）
            ng = gcost[cur] + movement_cost(gw, cur, nb)

            # 若邻居未访问过，或发现更优路径（新代价<原代价）
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng  # 更新实际代价
                parent[nb] = cur  # 记录父节点（用于路径重构）
                f = ng + goal_map.get(nb, gw.w + gw.h)  # 计算f值（实际代价+启发式距离）
                heapq.heappush(openpq, (f, nb))  # 邻居节点入队

    # 若队列耗尽仍未到达目标，返回(None, None)表示无路径
    return None, None


# --------- DP over subsets (Held–Karp) for collect-all-golds ---------
def best_collect_all_golds_dp(gw):
    """
    使用Held-Karp算法（子集动态规划）求解全金矿收集最优路径。

    核心思想：通过动态规划（DP）枚举所有金矿收集子集，计算从起点出发、收集指定子集金矿并结束于特定金矿的最小代价，
    最终找到收集所有金矿后到达终点的最优路径，适用于金矿数量较少（通常<15）的场景。

    参数:
        gw: GridWorld - 网格世界环境对象（包含金矿位置、起点、终点等信息）
    返回值:
        TrajectoryResult - 包含最优收集路径和总奖励的结果对象
    """
    # 提取所有金矿位置（单元格类型为'$'的坐标）
    golds = [p for p, t in gw.cells.items() if t == '$']
    s, g = gw.start, gw.goal  # s:起点坐标, g:终点坐标

    # 边界情况：若无金矿，直接返回起点到终点的最短路径
    if not golds:
        p = shortest_path(gw, s, g) or []
        return TrajectoryResult(p, total_reward_for_path(gw, p))

    # 预处理：为每个金矿分配索引（用于位掩码表示），K为金矿总数
    idx = {pos: i for i, pos in enumerate(golds)}
    K = len(golds)

    # 缓存：存储关键点对（起点/金矿/终点）之间的路径和代价
    PATH = {}  # PATH[(a, b)]: 从a到b的路径坐标列表
    COST = {}  # COST[(a, b)]: 从a到b的最小移动代价

    # 辅助函数：预计算两点间的路径和代价（避开其他金矿）
    def pair(a, b):
        """
        预计算并缓存从a到b的路径和代价，避开除b外的其他金矿（防止提前收集）。
        若路径不存在，尝试放宽避障条件（不避开任何金矿）。
        """
        if (a, b) in PATH:  # 已缓存，直接返回
            return
        # 避障集合：所有金矿中排除目标b（只允许经过目标金矿）
        avoid = set(golds) - {b}
        # 使用带避障的A*计算路径和代价
        p, c = a_star_movement(gw, a, b, avoid=avoid)
        # 若路径不存在（被其他金矿完全阻挡），放宽避障条件（不避开任何金矿）
        if p is None:
            p, c = a_star_movement(gw, a, b, avoid=set())
        # 缓存路径（空路径用[]表示）和代价（不可达用极大值1e9表示）
        PATH[(a, b)] = p or []
        COST[(a, b)] = c if c is not None else 10 ** 9

    # 批量预计算所有关键点对的路径和代价：
    # 关键点包括：起点(s)、所有金矿(golds)、终点(g)
    for a in [s] + golds:  # 起点 + 每个金矿作为起点
        for b in golds + [g]:  # 每个金矿 + 终点作为终点
            if a != b:  # 跳过起点和终点相同的情况
                pair(a, b)

    # -------------------------- 动态规划核心 --------------------------
    # DP状态定义：dp[mask][i] = 从起点出发，收集mask中所有金矿，最终到达第i个金矿的最小移动代价
    #   - mask: 位掩码（K位二进制数，第j位为1表示第j个金矿已收集）
    #   - i: 当前结束的金矿索引（0~K-1）
    INF = 10 ** 9  # 无穷大（表示不可达状态）
    dp = [[INF] * K for _ in range(1 << K)]  # DP表：(2^K)行 × K列
    pre = [[None] * K for _ in range(1 << K)]  # 前驱指针表：记录dp[mask][i]的最优前驱金矿索引

    # 初始化DP表：仅包含单个金矿的子集（mask为只有第i位为1的状态）
    for i, gi in enumerate(golds):
        # 从起点(s)到第i个金矿(gi)的代价作为初始状态
        dp[1 << i][i] = COST[(s, gi)]
        pre[1 << i][i] = -1  # 前驱标记为-1，表示从起点直接到达

    # 填充DP表：遍历所有可能的金矿子集（mask）
    for mask in range(1 << K):
        # 遍历当前子集中已收集的金矿i（作为终点）
        for i in range(K):
            # 若mask中不包含第i个金矿，跳过
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask][i]
            # 若当前状态不可达（代价为INF），跳过
            if current_cost >= INF:
                continue
            current_gold = golds[i]  # 当前终点金矿的坐标

            # 尝试添加未收集的金矿j（扩展子集）
            for j in range(K):
                # 若mask中已包含第j个金矿，跳过
                if mask & (1 << j):
                    continue
                next_gold = golds[j]  # 待添加的金矿坐标
                new_mask = mask | (1 << j)  # 新子集：mask | {j}（添加第j个金矿）
                # 新代价 = 当前代价 + 从金矿i到金矿j的预计算代价
                candidate_cost = current_cost + COST[(current_gold, next_gold)]

                # 若新代价优于现有状态，更新DP表和前驱指针
                if candidate_cost < dp[new_mask][j]:
                    dp[new_mask][j] = candidate_cost
                    pre[new_mask][j] = i  # 记录前驱为i

    # -------------------------- 计算最终路径 --------------------------
    # 1. 计算收集所有金矿后到达终点的最小总代价
    full_mask = (1 << K) - 1  # 全1掩码，表示所有金矿已收集
    best_total_cost = INF
    best_last_gold = None  # 最优路径中最后一个收集的金矿索引
    for i in range(K):
        # 总代价 = 收集所有金矿到i的代价 + 从i到终点g的代价
        total_cost = dp[full_mask][i] + COST[(golds[i], g)]
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_gold = i

    # 2. 回溯重构金矿收集顺序（从最后一个金矿反向追溯）
    order = []
    current_mask = full_mask
    current_i = best_last_gold
    # 若最优路径不存在（所有金矿不可达终点），使用贪心算法作为 fallback
    if current_i is None:
        return best_collect_all_golds_then_exit(gw)

    # 从全掩码状态开始回溯，直到mask为空
    while current_i is not None and current_i >= 0:
        order.append(current_i)  # 记录当前金矿索引
        prev_i = pre[current_mask][current_i]  # 获取前驱金矿索引
        current_mask ^= (1 << current_i)  # 移除当前金矿的掩码位
        current_i = prev_i
    # 反转顺序，得到从起点出发的金矿收集顺序（[第一个收集, ..., 最后一个收集]）
    order = list(reversed(order))

    # 3. 拼接完整路径：起点 → 按order收集金矿 → 终点
    # 路径序列：[起点, 金矿order[0], 金矿order[1], ..., 金矿order[-1], 终点]
    seq = [s] + [golds[k] for k in order] + [g]
    full_path: List[Coord] = [seq[0]]  # 初始化路径（起点）

    # 拼接序列中相邻点的预计算路径段（跳过重复的起点）
    for a, b in zip(seq, seq[1:]):
        seg = PATH[(a, b)]  # 获取a到b的预计算路径段
        # 若路径段为空（a和b直接相邻或不可达），使用最短路径作为fallback
        if not seg:
            seg = shortest_path(gw, a, b) or [a, b]
        full_path.extend(seg[1:])  # 拼接路径段（跳过seg[0]避免与前一段终点重复）

    # 计算路径的总奖励（基于原奖励规则，用于结果报告）
    score = total_reward_for_path(gw, full_path)
    return TrajectoryResult(full_path, score)

# ------------------ Part 3: Dynamic programming & RL planning ------------------
@dataclass
class ValueIterationResult:
    values: Dict[Coord, float]
    policy: Dict[Coord, Optional[str]]
    path: List[Coord]
    iterations: int
    converged: bool


@dataclass
class PolicyIterationResult:
    values: Dict[Coord, float]
    policy: Dict[Coord, Optional[str]]
    path: List[Coord]
    policy_iterations: int
    evaluation_iterations: int


@dataclass
class QLearningResult:
    q_values: Dict[Tuple[Coord, str], float]
    policy: Dict[Coord, Optional[str]]
    path: List[Coord]
    episode_rewards: List[float]
    episodes: int


def _enumerate_states(gw):
    states: List[Coord] = []
    for r in range(gw.h):
        for c in range(gw.w):
            coord = (c, r)
            if gw.is_wall(coord):
                continue
            states.append(coord)
    return states


def _apply_action(gw, state, action, collected_golds=None):
    """
    应用动作并返回下一个状态、奖励和是否结束。
    
    参数:
        gw: GridWorld - 网格世界环境
        state: Coord - 当前状态（位置）
        action: str - 要执行的动作
        collected_golds: Optional[Set[Coord]] - 已收集的金币集合（用于避免重复奖励）
    
    返回:
        (下一个状态, 奖励, 是否结束)
    """
    if gw.is_terminal(state):
        return state, 0, True
    dx, dy = ACTIONS[action]
    nxt = (state[0] + dx, state[1] + dy)
    if not gw.in_bounds(nxt) or gw.is_wall(nxt):
        return state, gw.wall_penalty, False
    
    # 奖励函数设置
    cell_type = gw.cell_type(nxt)
    if cell_type == 'G':  # 目标（出口）
        reward = 20
    elif cell_type == 'B':  # 陷阱
        reward = -10
    elif cell_type == '$':  # 金币
        # 如果追踪金币收集（rollout阶段），检查是否已收集
        if collected_golds is not None:
            if nxt not in collected_golds:
                reward = 20  # 首次收集：+20
                collected_golds.add(nxt)  # 标记为已收集
            else:
                reward = -1  # 已收集过：按普通通道处理
        else:
            # 训练阶段（不追踪金币）：仍给20奖励
            reward = 20
    elif cell_type == 'S':  # 入口
        reward = -1
    elif cell_type == '#':
        reward = -5
    else:  # 通道(P)
        reward = -1
    
    done = gw.is_terminal(nxt)
    return nxt, reward, done


def _best_action(
    gw, state, values, gamma
):
    if gw.is_terminal(state):
        return None, 0.0
    best_action: Optional[str] = None
    best_value = float('-inf')
    best_next = state
    for action in ACTION_ORDER:
        nxt, reward, done = _apply_action(gw, state, action)
        cand = reward + (0.0 if done else gamma * values[nxt])
        if cand > best_value:
            best_value = cand
            best_action = action
            best_next = nxt
        elif cand == best_value and best_action is not None:
            # 同分时优先不走陷阱，其次比较到目标的最短步数距离
            t_cur = gw.cell_type(nxt)
            t_best = gw.cell_type(best_next)
            if t_best == 'B' and t_cur != 'B':
                best_value = cand
                best_action = action
                best_next = nxt
            else:
                cur_dist = grid_steps_dist(gw, nxt, gw.goal)
                best_dist = grid_steps_dist(gw, best_next, gw.goal)
                if nxt != state and (best_next == state or cur_dist < best_dist):
                    best_value = cand
                    best_action = action
                    best_next = nxt
    return best_action, best_value if best_value != float('-inf') else 0.0


def _rollout_policy(gw, policy, max_steps = 200):
    """
    执行策略生成路径，追踪已收集的金币以避免无限循环。
    
    参数:
        gw: GridWorld - 网格世界环境
        policy: Dict[Coord, str] - 策略字典
        max_steps: int - 最大步数
    
    返回:
        List[Coord] - 生成的路径
    """
    state = gw.start
    path: List[Coord] = [state]
    collected_golds: Set[Coord] = set()  # 追踪已收集的金币
    visited_count: Dict[Coord, int] = {state: 1}
    
    for _ in range(max_steps):
        action = policy.get(state)
        if action is None:
            break

        # 首先按策略动作前进
        nxt, _, done = _apply_action(gw, state, action, collected_golds)

        # 如出现回退、在某格子访问次数过多，或下一步是陷阱，尝试改用更接近终点的备选动作
        need_escape = False
        if len(path) >= 2 and nxt == path[-2]:
            need_escape = True
        if visited_count.get(nxt, 0) >= 2:
            need_escape = True
        if gw.cell_type(nxt) == 'B':
            need_escape = True

        if need_escape:
            best_alt = None
            best_alt_dist = None
            for a in ACTION_ORDER:
                if a == action:
                    continue
                dx, dy = ACTIONS[a]
                cand = (state[0] + dx, state[1] + dy)
                if not gw.in_bounds(cand) or gw.is_wall(cand):
                    continue
                if visited_count.get(cand, 0) >= 2:
                    continue
                # 避开陷阱作为备选
                if gw.cell_type(cand) == 'B':
                    continue
                d = grid_steps_dist(gw, cand, gw.goal)
                if best_alt is None or d < best_alt_dist:
                    best_alt = a
                    best_alt_dist = d
            if best_alt is not None:
                # 使用备选动作重新计算下一步
                nxt, _, done = _apply_action(gw, state, best_alt, collected_golds)

        path.append(nxt)
        state = nxt
        
        # 检测是否到达终点
        if done:
            break
        
        # 更新访问计数，用于抑制小圈来回
        visited_count[nxt] = visited_count.get(nxt, 0) + 1
    
    return path


def value_iteration_plan(
    gw,
    gamma = 0.9,
    theta = 1e-4,
    max_iterations = 1000,
):
    states = _enumerate_states(gw)
    values = {s: 0.0 for s in states}
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        delta = 0.0
        new_values = values.copy()
        for state in states:
            if gw.is_terminal(state):
                new_values[state] = 0.0
                continue
            # 贝尔曼最优方程：V(s) = max_a [ R(s,a) + γ * V(s') ]
            best_v = float('-inf')
            for action in ACTION_ORDER:
                nxt, reward, done = _apply_action(gw, state, action)
                val = reward + (0.0 if done else gamma * values[nxt])
                if val > best_v:
                    best_v = val
            if best_v == float('-inf'):
                best_v = 0.0
            new_values[state] = best_v
            delta = max(delta, abs(new_values[state] - values[state]))
        values = new_values
        if delta < theta:
            converged = True
            break
    policy = {}
    for state in states:
        if gw.is_terminal(state):
            policy[state] = None
            continue
        # 政策由贝尔曼最优选择决定
        best_action = None
        best_v = float('-inf')
        for action in ACTION_ORDER:
            nxt, reward, done = _apply_action(gw, state, action)
            val = reward + (0.0 if done else gamma * values[nxt])
            if val > best_v:
                best_v = val
                best_action = action
            elif val == best_v and best_action is not None:
                # 同分时优先不走陷阱，其次选更靠近目标的动作
                t_cur = gw.cell_type(nxt)
                best_next_state = _apply_action(gw, state, best_action)[0]
                t_best = gw.cell_type(best_next_state)
                if t_best == 'B' and t_cur != 'B':
                    best_action = action
                else:
                    cur_dist = grid_steps_dist(gw, nxt, gw.goal)
                    best_dist = grid_steps_dist(gw, best_next_state, gw.goal)
                    if nxt != state and (best_dist == 0 or cur_dist < best_dist):
                        best_action = action
        policy[state] = best_action
    path = _rollout_policy(gw, policy)
    return ValueIterationResult(values, policy, path, iterations, converged)


def policy_iteration_plan(
    gw,
    gamma = 0.9,
    theta = 1e-4,
    max_policy_iterations = 100,
):
    states = _enumerate_states(gw)
    policy = {}
    for state in states:
        policy[state] = None if gw.is_terminal(state) else ACTION_ORDER[0]
    values = {s: 0.0 for s in states}
    policy_iters = 0
    evaluation_iters_total = 0
    for policy_iters in range(1, max_policy_iterations + 1):
        # Policy evaluation (迭代求解 V(s) = R(s,pi(s)) + γ V(s') )
        while True:
            delta = 0.0
            evaluation_iters_total += 1
            for state in states:
                if gw.is_terminal(state) or policy[state] is None:
                    new_val = 0.0
                else:
                    nxt, reward, done = _apply_action(gw, state, policy[state])
                    new_val = reward + (0.0 if done else gamma * values[nxt])
                delta = max(delta, abs(new_val - values[state]))
                values[state] = new_val
            if delta < theta:
                break
        # Policy improvement (贝尔曼最优式)
        policy_stable = True
        for state in states:
            if gw.is_terminal(state):
                policy[state] = None
                continue
            old_action = policy[state]
            best_action = None
            best_v = float('-inf')
            for action in ACTION_ORDER:
                nxt, reward, done = _apply_action(gw, state, action)
                val = reward + (0.0 if done else gamma * values[nxt])
                if val > best_v:
                    best_v = val
                    best_action = action
                elif val == best_v and best_action is not None:
                    # 同分时优先不走陷阱，其次选更靠近目标的动作
                    t_cur = gw.cell_type(nxt)
                    best_next_state = _apply_action(gw, state, best_action)[0]
                    t_best = gw.cell_type(best_next_state)
                    if t_best == 'B' and t_cur != 'B':
                        best_action = action
                    else:
                        cur_dist = grid_steps_dist(gw, nxt, gw.goal)
                        best_dist = grid_steps_dist(gw, best_next_state, gw.goal)
                        if nxt != state and (best_dist == 0 or cur_dist < best_dist):
                            best_action = action
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    path = _rollout_policy(gw, policy)
    return PolicyIterationResult(values, policy, path, policy_iters, evaluation_iters_total)


def q_learning_plan(
    gw,
    episodes = 500,
    gamma = 0.9,
    alpha = 0.1,
    epsilon = 0.1,
    epsilon_decay = 0.995,
    min_epsilon = 0.01,
    max_steps_per_episode = 200,
    seed = None,
) -> QLearningResult:
    rng = random.Random(seed)
    q_values: DefaultDict[Tuple[Coord, str], float] = defaultdict(float)
    episode_rewards: List[float] = []
    states = _enumerate_states(gw)
    for ep in range(episodes):
        state = gw.start
        total_reward = 0.0
        for _ in range(max_steps_per_episode):
            if gw.is_terminal(state):
                break
            if rng.random() < epsilon:
                action = rng.choice(ACTION_ORDER)
            else:
                action = max(
                    ACTION_ORDER,
                    key=lambda a: (q_values[(state, a)], -grid_steps_dist(gw, _apply_action(gw, state, a)[0], gw.goal))
                )
            nxt, reward, done = _apply_action(gw, state, action)
            best_next = 0.0
            if not done:
                best_next = max(q_values[(nxt, a)] for a in ACTION_ORDER)
            old_q = q_values[(state, action)]
            q_values[(state, action)] = (1 - alpha) * old_q + alpha * (reward + gamma * best_next)
            state = nxt
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    policy: Dict[Coord, Optional[str]] = {}
    for state in states:
        if gw.is_terminal(state):
            policy[state] = None
        else:
            best_action = max(
                ACTION_ORDER,
                key=lambda a: (q_values[(state, a)], -grid_steps_dist(gw, _apply_action(gw, state, a)[0], gw.goal))
            )
            policy[state] = best_action
    path = _rollout_policy(gw, policy)
    return QLearningResult(dict(q_values), policy, path, episode_rewards, episodes)
