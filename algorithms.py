from __future__ import annotations
from typing import Tuple, Dict, List, Optional, Set, Mapping, DefaultDict
import heapq, itertools, random
from collections import defaultdict

from dataclasses import dataclass, field

from grid import GridWorld, Coord

# ------------------ Utilities ------------------
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


ACTIONS: Dict[str, Tuple[int, int]] = {
    'U': (0, -1),
    'R': (1, 0),
    'D': (0, 1),
    'L': (-1, 0),
}
ACTION_ORDER: List[str] = ['U', 'R', 'D', 'L']

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

# ------------------ Demon escape via minimax ------------------
@dataclass
class MinimaxEscapeResult:
    agent_path: List[Coord]
    demon_path: List[Coord]
    success: bool
    demon_start: Optional[Coord]
    caught_step: Optional[int]
    score: int


def minimax_escape(
    gw: GridWorld,
    depth: int = 6,
    max_steps: int = 80,
    demon_start: Optional[Coord] = None,
    seed: Optional[int] = None,
) -> MinimaxEscapeResult:
    """Simulate an escape attempt using minimax planning against a roaming demon."""

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
    opening_score: Optional[int] = None
    caught_step: Optional[int] = None

    WIN_SCORE = 1000
    LOSE_SCORE = -1000

    def heuristic(a: Coord, d: Optional[Coord]) -> int:
        dist_goal = manhattan(a, start_goal)
        dist_demon = manhattan(a, d) if d is not None else gw.w + gw.h
        return -10 * dist_goal + 3 * dist_demon

    def minimax(
        agent: Coord,
        demon: Coord,
        remaining: int,
        maximizing: bool,
        alpha: float,
        beta: float,
        memo: Dict[Tuple[Coord, Coord, int, bool], Tuple[int, Optional[Coord]]],
    ) -> Tuple[int, Optional[Coord]]:
        key = (agent, demon, remaining, maximizing)
        if key in memo:
            return memo[key]
        if agent == demon:
            val = LOSE_SCORE + remaining
            memo[key] = (val, None)
            return memo[key]
        if agent == start_goal:
            val = WIN_SCORE - remaining
            memo[key] = (val, None)
            return memo[key]
        if remaining == 0:
            val = heuristic(agent, demon)
            memo[key] = (val, None)
            return memo[key]

        if maximizing:
            value = float('-inf')
            best_move: Optional[Coord] = None
            moves = gw.neighbors4(agent)
            if not moves:
                moves = [agent]
            for nxt in moves:
                if nxt == demon:
                    cand = LOSE_SCORE + remaining - 1
                else:
                    cand = minimax(nxt, demon, remaining - 1, False, alpha, beta, memo)
                value = max(value, cand)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            memo[key] = int(value)
            return memo[key]
        else:
            value = float('inf')
            moves = gw.neighbors4(demon)
            if not moves:
                moves = [demon]
            for nxt in moves:
                if nxt == agent:
                    cand = LOSE_SCORE + remaining - 1
                else:
                    cand = minimax(agent, nxt, remaining - 1, True, alpha, beta, memo)
                value = min(value, cand)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            memo[key] = int(value)
            return memo[key]

    def choose_best(agent: Coord, demon: Coord) -> Tuple[int, Optional[Coord]]:
        memo: Dict[Tuple[Coord, Coord, int, bool], int] = {}
        best_val = float('-inf')
        best_move: Optional[Coord] = None
        moves = gw.neighbors4(agent)
        if not moves:
            moves = [agent]
        for nxt in moves:
            if nxt == demon:
                cand = LOSE_SCORE + depth - 1
            else:
                cand = minimax(nxt, demon, depth - 1, False, float('-inf'), float('inf'), memo)
            prefer = True if best_move is None else manhattan(nxt, start_goal) < manhattan(best_move, start_goal)
            if cand > best_val or (cand == best_val and prefer):
                best_val = cand
                best_move = nxt
        return int(best_val), best_move

    for step in range(max_steps):
        if agent_pos == start_goal:
            break
        if demon_pos is not None and agent_pos == demon_pos:
            caught_step = step
            break
        if demon_pos is None:
            demon_path = []
        score, move = choose_best(agent_pos, demon_pos) if demon_pos is not None else (WIN_SCORE, None)
        if opening_score is None:
            opening_score = score
        if move is None:
            break
        agent_pos = move
        agent_path.append(agent_pos)
        if demon_pos is not None:
            if agent_pos == demon_pos:
                caught_step = step + 1
                break
            demon_moves = gw.neighbors4(demon_pos)
            if not demon_moves:
                demon_moves = [demon_pos]
            demon_pos = rng.choice(demon_moves)
            demon_path.append(demon_pos)
            if agent_pos == demon_pos:
                caught_step = step + 1
                break
        if agent_pos == start_goal:
            break

    success = agent_path[-1] == start_goal and (demon_pos is None or agent_path[-1] != demon_pos)
    final_score = opening_score if opening_score is not None else heuristic(agent_path[-1], demon_pos)
    return MinimaxEscapeResult(agent_path, demon_path, success, demon_start or gw.demon, caught_step, final_score)

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


def _enumerate_states(gw: GridWorld) -> List[Coord]:
    states: List[Coord] = []
    for r in range(gw.h):
        for c in range(gw.w):
            coord = (c, r)
            if gw.is_wall(coord):
                continue
            states.append(coord)
    return states


def _apply_action(gw: GridWorld, state: Coord, action: str) -> Tuple[Coord, int, bool]:
    if gw.is_terminal(state):
        return state, 0, True
    dx, dy = ACTIONS[action]
    nxt = (state[0] + dx, state[1] + dy)
    if not gw.in_bounds(nxt) or gw.is_wall(nxt):
        return state, gw.wall_penalty, False
    reward = gw.reward_for(nxt)
    done = gw.is_terminal(nxt)
    return nxt, reward, done


def _best_action(
    gw: GridWorld,
    state: Coord,
    values: Mapping[Coord, float],
    gamma: float,
) -> Tuple[Optional[str], float]:
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
            # Tie-breaker: prefer moves closer to goal and avoid standing still
            cur_dist = manhattan(nxt, gw.goal)
            best_dist = manhattan(best_next, gw.goal)
            if nxt != state and (best_next == state or cur_dist < best_dist):
                best_value = cand
                best_action = action
                best_next = nxt
    return best_action, best_value if best_value != float('-inf') else 0.0


def _rollout_policy(gw: GridWorld, policy: Mapping[Coord, Optional[str]], max_steps: int = 200) -> List[Coord]:
    state = gw.start
    path: List[Coord] = [state]
    for _ in range(max_steps):
        action = policy.get(state)
        if action is None:
            break
        nxt, _, done = _apply_action(gw, state, action)
        path.append(nxt)
        state = nxt
        if done or nxt == path[-2]:
            break
    return path


def value_iteration_plan(
    gw: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-4,
    max_iterations: int = 1000,
) -> ValueIterationResult:
    states = _enumerate_states(gw)
    values: Dict[Coord, float] = {s: 0.0 for s in states}
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        delta = 0.0
        for state in states:
            if gw.is_terminal(state):
                values[state] = 0.0
                continue
            _, best_value = _best_action(gw, state, values, gamma)
            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
        if delta < theta:
            converged = True
            break
    policy: Dict[Coord, Optional[str]] = {}
    for state in states:
        action, _ = _best_action(gw, state, values, gamma)
        policy[state] = action
    path = _rollout_policy(gw, policy)
    return ValueIterationResult(values, policy, path, iterations, converged)


def policy_iteration_plan(
    gw: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-4,
    max_policy_iterations: int = 100,
) -> PolicyIterationResult:
    states = _enumerate_states(gw)
    policy: Dict[Coord, Optional[str]] = {}
    for state in states:
        policy[state] = None if gw.is_terminal(state) else ACTION_ORDER[0]
    values: Dict[Coord, float] = {s: 0.0 for s in states}
    policy_iters = 0
    evaluation_iters_total = 0
    for policy_iters in range(1, max_policy_iterations + 1):
        # Policy evaluation
        while True:
            delta = 0.0
            evaluation_iters_total += 1
            for state in states:
                if gw.is_terminal(state) or policy[state] is None:
                    values[state] = 0.0
                    continue
                nxt, reward, done = _apply_action(gw, state, policy[state])
                new_val = reward + (0.0 if done else gamma * values[nxt])
                delta = max(delta, abs(new_val - values[state]))
                values[state] = new_val
            if delta < theta:
                break
        # Policy improvement
        policy_stable = True
        for state in states:
            if gw.is_terminal(state):
                policy[state] = None
                continue
            old_action = policy[state]
            best_action, _ = _best_action(gw, state, values, gamma)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    path = _rollout_policy(gw, policy)
    return PolicyIterationResult(values, policy, path, policy_iters, evaluation_iters_total)


def q_learning_plan(
    gw: GridWorld,
    episodes: int = 500,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    max_steps_per_episode: int = 200,
    seed: Optional[int] = None,
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
                    key=lambda a: (q_values[(state, a)], -manhattan(_apply_action(gw, state, a)[0], gw.goal))
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
                key=lambda a: (q_values[(state, a)], -manhattan(_apply_action(gw, state, a)[0], gw.goal))
            )
            policy[state] = best_action
    path = _rollout_policy(gw, policy)
    return QLearningResult(dict(q_values), policy, path, episode_rewards, episodes)


# ------------------ Part 1: Step-by-step search generators ------------------

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
