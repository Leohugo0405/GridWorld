from __future__ import annotations
from typing import Tuple, Dict, List, Optional, Set
import heapq, itertools
from dataclasses import dataclass, field

from grid import GridWorld, Coord

# ------------------ Utilities ------------------
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def reconstruct(parent: Dict[Coord, Optional[Coord]], t: Coord) -> List[Coord]:
    path = []
    while t is not None:
        path.append(t)
        t = parent[t]
    return list(reversed(path))

# ------------------ Part 1: Pathfinding ------------------
def dfs_find_path(gw: GridWorld, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    stack = [start]
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    seen: Set[Coord] = set()
    while stack:
        cur = stack.pop()
        if cur in seen: 
            continue
        seen.add(cur)
        if cur == goal:
            return reconstruct(parent, cur)
        for nb in gw.neighbors4(cur):
            if nb not in parent:
                parent[nb] = cur
                stack.append(nb)
    return None

def greedy_best_first(gw: GridWorld, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    pq: List[Tuple[int, Coord]] = []
    heapq.heappush(pq, (manhattan(start, goal), start))
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    seen: Set[Coord] = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return reconstruct(parent, cur)
        for nb in gw.neighbors4(cur):
            if nb not in parent:
                parent[nb] = cur
                heapq.heappush(pq, (manhattan(nb, goal), nb))
    return None

def a_star(gw: GridWorld, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    openpq: List[Tuple[int, Coord]] = []
    gcost: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    heapq.heappush(openpq, (manhattan(start, goal), start))
    while openpq:
        _, cur = heapq.heappop(openpq)
        if cur == goal:
            return reconstruct(parent, cur)
        for nb in gw.neighbors4(cur):
            ng = gcost[cur] + 1
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng
                parent[nb] = cur
                f = ng + manhattan(nb, goal)
                heapq.heappush(openpq, (f, nb))
    return None

def shortest_path(gw: GridWorld, s: Coord, t: Coord) -> Optional[List[Coord]]:
    return a_star(gw, s, t)

# ------------------ Part 2: Rewards + Collect-All-Golds ------------------
@dataclass
class TrajectoryResult:
    path: List[Coord]
    total_reward: int

def total_reward_for_path(gw: GridWorld, path: List[Coord]) -> int:
    if not path: return -10**9
    # step -1; stepping into trap -10; stepping into gold +10 (once); reaching G +20
    reward = 0
    taken: Set[Coord] = set()
    for i in range(len(path)-1):
        cur, nxt = path[i], path[i+1]
        reward -= 1  # step dissipation
        t = gw.cell_type(nxt)
        if t == 'B':
            reward += -10
        if t == '$' and nxt not in taken:
            reward += 10
            taken.add(nxt)
    if path[-1] == gw.goal:
        reward += 20
    return reward

def best_collect_all_golds_then_exit(gw: GridWorld, max_perm: int = 7) -> TrajectoryResult:
    golds = [p for p,t in gw.cells.items() if t == '$']
    s, g = gw.start, gw.goal
    if not golds:
        p = shortest_path(gw, s, g) or []
        return TrajectoryResult(p, total_reward_for_path(gw, p))

    # precompute pairwise shortest paths
    points = [s] + golds + [g]
    sp_cache: Dict[Tuple[Coord,Coord], List[Coord]] = {}
    for a, b in itertools.permutations(points, 2):
        if (a,b) not in sp_cache:
            sp_cache[(a,b)] = shortest_path(gw, a, b) or []

    best_path: List[Coord] = []
    best_score = -10**9

    if len(golds) <= max_perm:
        for perm in itertools.permutations(golds):
            seq = [s] + list(perm) + [g]
            path: List[Coord] = [seq[0]]
            ok = True
            for i in range(len(seq)-1):
                seg = sp_cache.get((seq[i], seq[i+1])) or []
                if not seg:
                    ok = False; break
                path.extend(seg[1:])
            if not ok: 
                continue
            score = total_reward_for_path(gw, path)
            if score > best_score:
                best_score, best_path = score, path
    else:
        # greedy heuristic: choose gold maximizing (10 - dist)
        remaining = set(golds)
        cur = s
        path: List[Coord] = [cur]
        while remaining:
            target = max(remaining, key=lambda z: (10 - manhattan(cur, z)))
            seg = sp_cache.get((cur, target)) or shortest_path(gw, cur, target) or []
            if not seg: break
            path.extend(seg[1:])
            cur = target
            remaining.remove(target)
        seg = sp_cache.get((cur, g)) or shortest_path(gw, cur, g) or []
        path.extend(seg[1:])
        best_path = path
        best_score = total_reward_for_path(gw, best_path)

    return TrajectoryResult(best_path, best_score)

# --------- Movement-cost A* with optional gold avoidance (for DP) ---------
def movement_cost(gw: GridWorld, from_p: Coord, to_p: Coord) -> int:
    # positive costs: 1 per step; entering trap adds 10
    c = 1
    tp = gw.cell_type(to_p)
    if tp == 'B':
        c += 10
    if tp == '#':
        c += 5
    if tp == 'P':
        c += 1
    return c

def a_star_movement(gw: GridWorld, start: Coord, goal: Coord, avoid: Optional[Set[Coord]]=None):
    avoid = avoid or set()
    openpq: List[Tuple[int, Coord]] = []
    gcost: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    heapq.heappush(openpq, (manhattan(start, goal), start))
    while openpq:
        _, cur = heapq.heappop(openpq)
        if cur == goal:
            return reconstruct(parent, cur), gcost[cur]
        for nb in gw.neighbors4(cur):
            if nb in avoid and nb != goal:
                continue
            ng = gcost[cur] + movement_cost(gw, cur, nb)
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng
                parent[nb] = cur
                f = ng + manhattan(nb, goal)
                heapq.heappush(openpq, (f, nb))
    return None, None

# --------- DP over subsets (Held–Karp) for collect-all-golds ---------
def best_collect_all_golds_dp(gw: GridWorld) -> TrajectoryResult:
    golds = [p for p,t in gw.cells.items() if t == '$']
    s, g = gw.start, gw.goal
    if not golds:
        p = shortest_path(gw, s, g) or []
        return TrajectoryResult(p, total_reward_for_path(gw, p))

    # Precompute pairwise paths & movement costs, avoiding other golds
    idx = {pos:i for i,pos in enumerate(golds)}
    K = len(golds)

    # Store path and cost between any two of (S + golds + G)
    PATH = {}
    COST = {}

    # Helper to compute path/cost avoiding other golds
    def pair(a, b):
        if (a,b) in PATH: return
        avoid = set(golds) - {b}
        p, c = a_star_movement(gw, a, b, avoid=avoid)
        if p is None:
            # Relax avoidance if necessary (rare)
            p, c = a_star_movement(gw, a, b, avoid=set())
        PATH[(a,b)] = p or []
        COST[(a,b)] = c if c is not None else 10**9

    # Fill caches
    for a in [s] + golds:
        for b in golds + [g]:
            if a != b:
                pair(a,b)

    # DP: dp[mask][i] = min movement cost to start at S, visit 'mask', end at gold i
    INF = 10**9
    dp = [[INF]*K for _ in range(1<<K)]
    pre = [[None]*K for _ in range(1<<K)]  # backpointer (prev_i)

    for i, gi in enumerate(golds):
        dp[1<<i][i] = COST[(s, gi)]
        pre[1<<i][i] = -1  # from start

    for mask in range(1<<K):
        for i in range(K):
            if not (mask & (1<<i)): continue
            ci = dp[mask][i]
            if ci >= INF: continue
            ai = golds[i]
            for j in range(K):
                if mask & (1<<j): continue
                aj = golds[j]
                newmask = mask | (1<<j)
                cand = ci + COST[(ai, aj)]
                if cand < dp[newmask][j]:
                    dp[newmask][j] = cand
                    pre[newmask][j] = i

    # Finish to goal
    full = (1<<K) - 1
    best_cost = INF
    best_last = None
    for i in range(K):
        cand = dp[full][i] + COST[(golds[i], g)]
        if cand < best_cost:
            best_cost = cand
            best_last = i

    # Reconstruct order of golds
    order = []
    mask = full
    i = best_last
    if i is None:
        # fallback
        return best_collect_all_golds_then_exit(gw)
    while i is not None and i >= 0:
        order.append(i)
        i_prev = pre[mask][i]
        mask ^= (1<<i)
        i = i_prev
    order = list(reversed(order))

    # Stitch full path: S -> golds in order -> G
    seq = [s] + [golds[k] for k in order] + [g]
    full_path: List[Coord] = [seq[0]]
    for a,b in zip(seq, seq[1:]):
        seg = PATH[(a,b)]
        if not seg:
            # fallback to shortest structural path
            seg = shortest_path(gw, a, b) or [a,b]
        full_path.extend(seg[1:])

    # Compute total reward under original reward scheme for reporting
    score = total_reward_for_path(gw, full_path)
    return TrajectoryResult(full_path, score)


# ------------------ Part 1: Step-by-step search generators ------------------
from dataclasses import dataclass

@dataclass
class SearchStep:
    algo: str
    current: Optional[Coord]
    frontier: List[Coord]
    visited: Set[Coord]
    parent: Dict[Coord, Optional[Coord]]

def dfs_search_steps(gw: GridWorld, start: Coord, goal: Coord):
    """Yield SearchStep snapshots while DFS explores; final yield includes full parent map."""
    stack = [start]
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    seen: Set[Coord] = set()
    yield SearchStep("DFS", None, list(stack), set(), dict(parent))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        yield SearchStep("DFS", cur, list(stack), set(seen), dict(parent))
        if cur == goal:
            break
        for nb in gw.neighbors4(cur):
            if nb not in parent:
                parent[nb] = cur
                stack.append(nb)
                yield SearchStep("DFS", cur, list(stack), set(seen), dict(parent))
    yield SearchStep("DFS", goal if goal in parent else cur, list(stack), set(seen), dict(parent))

def greedy_search_steps(gw: GridWorld, start: Coord, goal: Coord):
    pq: List[Tuple[int, Coord]] = []
    heapq.heappush(pq, (manhattan(start, goal), start))
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    seen: Set[Coord] = set()
    yield SearchStep("Greedy", None, [start], set(), dict(parent))
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        frontier_list = [c for _,c in pq]
        yield SearchStep("Greedy", cur, frontier_list, set(seen), dict(parent))
        if cur == goal:
            break
        for nb in gw.neighbors4(cur):
            if nb not in parent:
                parent[nb] = cur
                heapq.heappush(pq, (manhattan(nb, goal), nb))
                frontier_list = [c for _,c in pq]
                yield SearchStep("Greedy", cur, frontier_list, set(seen), dict(parent))
    yield SearchStep("Greedy", goal if goal in parent else cur, [c for _,c in pq], set(seen), dict(parent))

def astar_search_steps(gw: GridWorld, start: Coord, goal: Coord):
    openpq: List[Tuple[int, Coord]] = []
    gcost: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    heapq.heappush(openpq, (manhattan(start, goal), start))
    seen: Set[Coord] = set()
    yield SearchStep("A*", None, [start], set(), dict(parent))
    while openpq:
        _, cur = heapq.heappop(openpq)
        if cur in seen:
            continue
        seen.add(cur)
        frontier_list = [c for _,c in openpq]
        yield SearchStep("A*", cur, frontier_list, set(seen), dict(parent))
        if cur == goal:
            break
        for nb in gw.neighbors4(cur):
            ng = gcost[cur] + 1
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng
                parent[nb] = cur
                f = ng + manhattan(nb, goal)
                heapq.heappush(openpq, (f, nb))
                frontier_list = [c for _,c in openpq]
                yield SearchStep("A*", cur, frontier_list, set(seen), dict(parent))
    yield SearchStep("A*", goal if goal in parent else cur, [c for _,c in openpq], set(seen), dict(parent))
