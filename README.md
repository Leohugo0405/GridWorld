# GridWorld

一个使用 Python 和 Pygame 实现的网格世界算法演示项目，包含路径搜索、全金币收集、魔鬼逃生、值迭代、策略迭代和 Q-Learning。通过桌面窗口观察搜索前沿、已访问节点、智能体轨迹及算法统计，也可以直接导入算法模块进行实验。

本文依据仓库现有的 [grid.py](grid.py)、[algorithms.py](algorithms.py)、[visualize_pygame.py](visualize_pygame.py) 和 [map.json](map.json) 编写。界面上的模式名称、算法实际调用和不同模式的计分规则在下文分别说明。

## 目录

- [功能概览](#功能概览)
- [文件结构](#文件结构)
- [环境要求与安装](#环境要求与安装)
- [运行与界面操作](#运行与界面操作)
- [地图格式与坐标](#地图格式与坐标)
- [算法实现](#算法实现)
- [奖励与终止规则](#奖励与终止规则)
- [代码调用与示例输出](#代码调用与示例输出)
- [参数与返回结果](#参数与返回结果)
- [当前限制与常见问题](#当前限制与常见问题)

## 功能概览

| 界面模式             | 实际实现                                 | 展示内容                                |
| -------------------- | ---------------------------------------- | --------------------------------------- |
| `Part1: Search`      | DFS、Greedy、A* 分步搜索生成器           | 已访问格、搜索前沿、当前节点及回溯路径  |
| `Part2: Collect-All` | 带移动代价的 A* + Held–Karp 子集动态规划 | 收集金币后前往出口的智能体动画          |
| `Part3: Minimax`     | **界面实际调用 Expectimax**              | 智能体与随机移动魔鬼的逃生轨迹          |
| `Part4: Value Iter`  | 随机转移模型上的值迭代                   | 路径、迭代次数、收敛标记                |
| `Part5: Policy Iter` | 随机转移模型上的策略迭代                 | 路径、策略改进次数、累计策略评估次数    |
| `Part6: Q-Learning`  | 表格型 Q-Learning，采用 ε-greedy 探索    | 路径、训练回合数、最后 100 回合平均奖励 |

此外，代码还提供独立的带 α–β 剪枝的 `minimax_escape()`，以及基于全排列/贪心的金币收集函数。它们不是对应界面模式当前直接调用的实现。

## 文件结构

```text
GridWorld/
├── README.md               # 本文档
├── grid.py                 # 网格、地图加载、格子类型、邻居与魔鬼初始位置
├── algorithms.py           # 搜索、金币收集、逃生、MDP 与强化学习算法
├── visualize_pygame.py     # 桌面入口、下拉菜单、动画和底部统计栏
├── map.json                # 默认 10 × 10 地图
├── .gitignore
└── assets/
    ├── README.txt          # 较早的素材说明；部分符号/文件名与当前代码不同
    ├── agent.png           # 智能体
    ├── demon.png           # 魔鬼
    ├── start.png           # 起点 S
    ├── goal.png            # 出口 G
    ├── trap.png            # 陷阱 B
    ├── gold.png            # 金币 $
    ├── wall.png            # 墙 #
    ├── plain.png           # 通道 P
    ├── obstacle.png        # 当前图片加载器未引用
    └── legend.png          # 当前图片加载器未引用
```

仓库目前没有 `requirements.txt`、`pyproject.toml`、安装包配置或自动化测试套件。运行方式是直接执行 Python 脚本。

## 环境要求与安装

- Python 3。源码使用 `dataclasses` 和延迟求值类型注解，因此语言特性至少需要 Python 3.7；仓库没有声明完整的版本兼容范围。
- Pygame：**图形界面唯一直接导入的第三方库**。`grid.py` 和 `algorithms.py` 仅依赖 Python 标准库，不需要 NumPy、PyTorch 或 Gym。
- 运行界面需要可用的桌面显示环境。
- Git 用于克隆仓库，也可以下载仓库 ZIP 后解压。

文档中的算法示例已在 Python 3.8.6 下验证；这不代表所有 Python/Pygame 组合都经过测试。仓库没有锁定 Pygame 版本。

### 1. 获取代码

```bash
git clone https://github.com/Leohugo0405/GridWorld.git
cd GridWorld
python -m venv .venv
```

如果系统的 Python 3 命令是 `python3`，创建环境时使用 `python3 -m venv .venv`。

### 2. 激活虚拟环境

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows 命令提示符：

```bat
.venv\Scripts\activate.bat
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 3. 安装界面依赖并启动

```bash
python -m pip install pygame
python visualize_pygame.py
```

请从仓库根目录运行。`map.json` 和 `assets/` 都相对于**当前工作目录**读取。只调用算法时可以跳过 Pygame 的安装。

PowerShell 若不允许执行激活脚本，可直接使用虚拟环境解释器，无需更改系统执行策略：

```powershell
.\.venv\Scripts\python.exe -m pip install pygame
.\.venv\Scripts\python.exe visualize_pygame.py
```

## 运行与界面操作

启动后显示标题为 `GridWorld (Dropdown Menu)` 的窗口，默认处于搜索模式且暂停。默认地图下窗口尺寸为 642 × 718 像素，主循环帧率上限为 60 FPS。算法在点击开始时计算，之后播放已生成的轨迹或搜索快照。

### 一次完整演示

1. 点击 `Mode`，选择一个模式。
2. 对于 `Part1: Search`，再在 `Algorithm` 中选择 `DFS`、`Greedy` 或 `A*`；未选择时使用 DFS。
3. 点击 `Start/Pause`，计算结果并开始播放。
4. 在 `Controls` 中使用 `Speed +` 加快动画，或使用 `Speed -` 放慢动画。
5. 切换模式或使用 `Restart` 前，建议先暂停；随后选择模式/重启，必要时重新选择搜索算法，再点击开始。
6. 关闭窗口退出。

| 控件          | 当前行为                                                     |
| ------------- | ------------------------------------------------------------ |
| `Mode`        | 选择六种模式之一，重新读取地图并清空已有结果                 |
| `Algorithm`   | 选择 DFS、Greedy、A*；会切换至搜索模式                       |
| `Start/Pause` | 切换播放状态；从暂停转为播放时执行当前模式的准备逻辑         |
| `Restart`     | 重新加载地图、重建动画器并清空计算结果；保持当前播放标志     |
| `Speed + / -` | 每次减少/增加 20 ms 的动画步间隔；逃生动画下限 60 ms，其余下限 40 ms |

**播放细节：**当前 `Start/Pause` 不是所有模式都能原位置续播。重新开始时，搜索和策略动画会复位，逃生模式会重新随机生成魔鬼并计算；金币模式在路径已缓存时可以继续。模式切换和 `Restart` 不会自动把播放标志改为暂停，如果当时处于播放状态，可能需要先点击一次暂停，再点击开始，才能准备新结果。重建动画器也会恢复默认速度。

搜索模式的已访问格为浅蓝色、前沿为橙色、当前节点为红色，黑线表示当前节点的父指针回溯路径。金币收集动画在智能体**离开金币格**时把该格显示为通道。策略模式展示路径，不绘制完整的价值热力图或策略箭头。

底部的 `length`、`len` 和 `path_len` 均是**坐标点数量**，包含起点；移动次数通常为该数值减一。

## 地图格式与坐标

`GridWorld.from_json(path='map.json')` 读取 JSON 中的 `layout` 二维数组。外层是行，内层是列；算法坐标统一为 **`(col, row)`**，左上角为 `(0, 0)`。

| 符号 | 含义     | 是否可通过                                   |
| ---- | -------- | -------------------------------------------- |
| `S`  | 起点     | 是                                           |
| `G`  | 出口     | 是；在 MDP/RL 中为终止状态                   |
| `P`  | 普通通道 | 是                                           |
| `#`  | 墙       | 否                                           |
| `B`  | 陷阱     | 搜索/收集/逃生可经过；在 MDP/RL 中为终止状态 |
| `$`  | 金币     | 是                                           |

默认地图：10 列 × 10 行，起点 `(0, 0)`，出口 `(9, 9)`，包含 31 个墙格、16 个金币格和 5 个陷阱格。

可供自定义的小地图示例：

```json
{
  "width": 4,
  "height": 3,
  "layout": [
    ["S", "P", "$", "P"],
    ["P", "#", "B", "P"],
    ["P", "P", "P", "G"]
  ]
}
```

加载器实际从 `layout` 推导宽高，**不使用**顶层的 `width` 和 `height`。请提供非空、每行等长的布局，并明确放置一个 `S` 和一个 `G`；当前没有完整的地图校验。未知符号会作为普通通道处理。不要按旧的 `assets/README.txt` 将 `O` 或 `W` 当作当前地图的障碍符号。

图形入口固定调用 `GridWorld.from_json()`，没有启用命令行地图参数。因此，`python visualize_pygame.py other.json` 不会切换地图。要更换界面地图，可修改根目录 `map.json` 后重启程序；代码调用则可使用：

```python
from grid import GridWorld

gw = GridWorld.from_json("other.json")
```

地图尺寸改变后应重启窗口，因为窗口大小仅在启动时计算。

## 算法实现

### 1. 路径搜索

- `dfs_search_steps()`：使用栈进行深度优先搜索，不保证最短路径。邻居由 `neighbors4()` 按右、左、下、上生成，再依次入栈。
- `greedy_search_steps()`：优先扩展到目标的 BFS 最短步数距离较小的节点，不累计起点代价。
- `astar_search_steps()` / `a_star()`：使用 `f = g + h`，其中 `g` 为累计步数，`h` 为预先通过 BFS 计算的到目标最短步数距离。每次移动代价为 1。

这些函数通过 `neighbors4()` 排除墙和越界格。陷阱、金币的奖励不参与上述路径选择，陷阱也不会终止搜索。

虽然部分旧注释写着“曼哈顿距离”，当前 Greedy 和 A* 实际调用的是 BFS 距离表。`manhattan()` 仍存在，但并非这些搜索的当前启发函数。BFS 在这里是距离预计算工具，没有独立的 BFS 界面模式。

三个分步生成器返回 `SearchStep`，含算法名、当前节点、前沿、已访问集合和父指针。`a_star()` 直接返回包含起终点的路径，不可达时返回 `None`；`shortest_path()` 是它的封装。

### 2. 全金币收集

界面调用 `best_collect_all_golds_dp()`：

1. 提取所有金币位置。
2. 使用 `a_star_movement()` 预计算起点、金币和出口之间的路径及代价。优先避开其他金币；若不可达，则放宽该限制重算。
3. 用 Held–Karp 子集动态规划求金币访问顺序：`dp[mask][i]` 表示访问指定金币子集、最后停在第 `i` 个金币的最小累计预计算代价。
4. 回溯并拼接路径，最后用 `total_reward_for_path()` 计算报告中的奖励。

移动代价由 `movement_cost()` 决定：进入 `P` 为 2，进入 `B` 为 11，进入 `S`、`G`、`$` 为 1。函数虽定义了墙的代价，但路径搜索会先排除墙，所以墙仍不可通过。

**优化目标与报告奖励不同。**DP 最小化上述移动代价；奖励函数按每步扣 1、金币只奖励一次等规则计算。因此不能把结果描述为对实际总奖励的全局最优解。预计算路径可能因放宽限制而途经其他金币，进一步限制了“严格最优收集顺序”的解释。

设金币数为 K，DP 部分时间复杂度为 `O(K² × 2^K)`，空间复杂度为 `O(K × 2^K)`，另有点对寻路开销。当前实现没有根据金币数量自动切换为低内存算法；默认地图有 16 个金币，增大数量时需注意指数级增长。

独立函数 `best_collect_all_golds_then_exit(gw, max_perm=7)` 在金币数不超过阈值时枚举排列，否则采用最近金币贪心策略。DP 无法构造完整访问顺序时会调用它，但回退结果不保证收集全部金币或到达出口，使用自定义地图时应检查返回路径。

### 3. 魔鬼逃生：Minimax 与 Expectimax

- `minimax_escape()`：智能体最大化、魔鬼最小化，使用 α–β 剪枝及局部缓存进行有限深度搜索。
- `minimax_escape_expectimax()`：智能体最大化，魔鬼节点取所有可行移动价值的平均值。
- 两者在实际模拟中都让魔鬼通过 `rng.choice()` 随机选择可通行邻居。

界面 `Part3: Minimax` 调用的是 **`minimax_escape_expectimax(gw, depth=15, max_steps=1000, ...)`**。每次准备该模式都会重新选择魔鬼初始位置和随机种子。

魔鬼初始位置排除起点、出口和墙，但可以位于金币或陷阱格。智能体先移动，然后魔鬼移动；同格碰撞判定为被捕，模拟还包含位置交换检查。成功要求最终位于出口且未与魔鬼重合，模拟也可能因达到步数上限而停止。

两种算法以到出口/魔鬼的 BFS 距离和步数构造启发评分，胜负基础分为 `+1000` / `-1000`，不累计金币和陷阱奖励。返回的 `score` 通常是**首次决策的估值**，不是整条轨迹的奖励。有限深度搜索不保证每次逃生成功。

### 4. 值迭代与策略迭代

状态为所有非墙格的坐标，动作为 `R`、`D`、`L`、`U`。随机转移模型为：

- 80% 概率按指定方向移动；
- 10% 概率向该方向左侧偏移；
- 10% 概率向该方向右侧偏移；
- 碰墙或越界则留在原位置，获得墙惩罚。

值迭代同步更新状态价值，使用 `max_a E[R + gamma × V(s')]`，当最大价值变化小于 `theta` 或达到迭代上限时停止。策略迭代交替执行原地更新的策略评估和基于期望回报的策略改进，直到策略稳定或达到外层迭代上限。终止状态价值为 0，进入终止格的奖励在转移时给出。

### 5. Q-Learning

`q_learning_plan()` 从起点开始每个训练回合，用 ε-greedy 选择动作，并从同一随机转移模型采样。Q 表按以下规则更新：

```text
Q(s, a) ← (1 - alpha) × Q(s, a)
          + alpha × [reward + gamma × max_a' Q(s', a')]
```

转移结束时不加入下一状态的价值。每回合结束后衰减 ε，并保持不小于 `min_epsilon`。界面训练 500 回合，固定 `seed=42`；函数本身的默认种子为 `None`。

**策略可视化的解释：**以上三个规划/学习函数最终都调用 `_rollout_policy()` 生成展示路径。该函数按动作确定性移动，不再采样侧滑；遇到回退、重复访问或陷阱时，可能改用更接近出口的备选动作，默认最多走 200 步。因此动画不是训练过程中一次随机轨迹，也不是对原策略逐步不加修改的执行。

## 奖励与终止规则

项目包含多套计分逻辑，不能用一张统一奖励表解释全部模式。

| 情况               | `total_reward_for_path()`：金币收集报告              | `_apply_action()`：MDP/RL |
| ------------------ | ---------------------------------------------------- | ------------------------- |
| 移入普通通道或起点 | 每步 -1                                              | -1                        |
| 移入金币           | 每步 -1，首次到该金币额外 +10                        | **每次进入 +20**          |
| 移入陷阱           | 每步 -1，额外 -10；继续行走                          | -10，回合终止             |
| 到达出口           | 每步 -1；仅路径最后一点为出口时额外 +20              | +20，回合终止             |
| 碰墙或越界         | 搜索生成的路径会排除；此评分函数本身不校验路径合法性 | -5，原地不动，不终止      |

`GridWorld.reward_for_type()` 另提供基础查询，其中金币为 +10；MDP/RL 的 `_apply_action()` 自行定义金币为 +20，并未直接复用该基础查询。`GridWorld` 上的 `step_cost`、`rewards` 属性也不是统一生效的参数入口。

当前 `_apply_action()` 虽接受 `collected_golds` 参数，但没有使用它，状态中也不包含金币收集历史。因此 MDP/RL 可以反复进入金币格获取奖励，学到的策略可能偏向循环获利。较大的训练平均奖励不能直接解释为更高的到达出口成功率。

路径搜索只考虑可达性/步数；逃生只考虑碰撞、出口和启发评分，均不使用上表中的累计金币奖励。

## 代码调用与示例输出

以下示例只需要标准库。将代码保存为仓库根目录下的 `demo.py`，然后运行：

```bash
python demo.py
```

### 搜索、金币收集与策略规划

```python
from grid import GridWorld
from algorithms import (
    a_star,
    best_collect_all_golds_dp,
    value_iteration_plan,
    policy_iteration_plan,
    q_learning_plan,
)

gw = GridWorld.from_json("map.json")
path = a_star(gw, gw.start, gw.goal)
print(f"Map: {gw.w}x{gw.h}, start={gw.start}, goal={gw.goal}")
print(f"A*: steps={len(path) - 1 if path else 'unreachable'}")

collect = best_collect_all_golds_dp(gw)
print(f"Collect-All: path_len={len(collect.path)}, reward={collect.total_reward}")

vi = value_iteration_plan(gw)
print(f"Value Iteration: path_len={len(vi.path)} "
      f"iterations={vi.iterations} converged={vi.converged}")

pi = policy_iteration_plan(gw)
print(f"Policy Iteration: path_len={len(pi.path)} "
      f"policy_iters={pi.policy_iterations} eval_iters={pi.evaluation_iterations}")

ql = q_learning_plan(gw, seed=42)
recent = ql.episode_rewards[-100:]
average = sum(recent) / len(recent) if recent else 0.0
print(f"Q-Learning: path_len={len(ql.path)} "
      f"episodes={ql.episodes} avg_reward(last100)={average:.1f}")
```

使用仓库默认地图、Python 3.8.6 和上述参数实测输出：

```text
Map: 10x10, start=(0, 0), goal=(9, 9)
A*: steps=28
Collect-All: path_len=77, reward=104
Value Iteration: path_len=51 iterations=115 converged=True
Policy Iteration: path_len=51 policy_iters=7 eval_iters=325
Q-Learning: path_len=51 episodes=500 avg_reward(last100)=1403.0
```

此处金币收集路径经过了全部 16 个金币并到达出口；三个策略规划结果的展示路径也都到达了 `(9, 9)`。这些是当前地图与参数下的结果，不是任意自定义地图的保证。Q-Learning 的高回合奖励包含重复金币收益；修改地图、参数或运行时版本后，输出可能变化。

### 分步搜索数据

```python
from grid import GridWorld
from algorithms import astar_search_steps, reconstruct

gw = GridWorld.from_json("map.json")
last = None
for step in astar_search_steps(gw, gw.start, gw.goal):
    last = step  # 也可以在这里读取 current、frontier、visited 做自定义展示

if last is not None and gw.goal in last.parent:
    path = reconstruct(last.parent, gw.goal)
    print(f"A*: steps={len(path) - 1}, visited={len(last.visited)}")
else:
    print("Goal unreachable")
```

默认地图输出：

```text
A*: steps=28, visited=33
```

### 固定魔鬼位置的逃生实验

```python
from grid import GridWorld
from algorithms import minimax_escape_expectimax

gw = GridWorld.from_json("map.json")
result = minimax_escape_expectimax(
    gw, depth=4, max_steps=60, demon_start=(9, 6), seed=42
)
print(f"success={result.success}, path_len={len(result.agent_path)}, "
      f"caught_step={result.caught_step}, score={result.score}")
```

默认地图实测输出：

```text
success=True, path_len=29, caught_step=None, score=-44
```

该示例刻意使用较浅搜索深度以方便运行，与界面的 `depth=15` 不同。`score=-44` 是初始决策估值，与最终 `success=True` 并不矛盾。要比较 Minimax，可换用 `minimax_escape()` 并保留其他参数。

## 参数与返回结果

参数当前通过 Python 函数调用或修改源码设置，没有命令行参数解析和界面参数编辑器。

| 函数                                               | 主要默认参数                                            | 返回结果                                |
| -------------------------------------------------- | ------------------------------------------------------- | --------------------------------------- |
| `GridWorld.from_json()`                            | `path='map.json'`                                       | `GridWorld`                             |
| `a_star()` / `shortest_path()`                     | 显式传入地图、起点、目标                                | 路径列表或 `None`                       |
| `a_star_movement()`                                | `avoid=None`                                            | `(path, cost)`；不可达为 `(None, None)` |
| `best_collect_all_golds_dp()`                      | 无额外参数                                              | `TrajectoryResult(path, total_reward)`  |
| `best_collect_all_golds_then_exit()`               | `max_perm=7`                                            | `TrajectoryResult`                      |
| `minimax_escape()` / `minimax_escape_expectimax()` | `depth=10, max_steps=1000, demon_start=None, seed=None` | `MinimaxEscapeResult`                   |
| `value_iteration_plan()`                           | `gamma=0.9, theta=1e-4, max_iterations=1000`            | `ValueIterationResult`                  |
| `policy_iteration_plan()`                          | `gamma=0.9, theta=1e-4, max_policy_iterations=100`      | `PolicyIterationResult`                 |
| `q_learning_plan()`                                | 见下表                                                  | `QLearningResult`                       |

Q-Learning 默认参数：

| 参数                    | 默认值 | 含义                  |
| ----------------------- | ------ | --------------------- |
| `episodes`              | 500    | 训练回合数            |
| `gamma`                 | 0.9    | 折扣因子              |
| `alpha`                 | 0.1    | 学习率                |
| `epsilon`               | 0.1    | 初始随机探索概率      |
| `epsilon_decay`         | 0.995  | 每回合探索概率乘数    |
| `min_epsilon`           | 0.01   | 探索概率下限          |
| `max_steps_per_episode` | 200    | 每回合最大训练步数    |
| `seed`                  | `None` | 随机种子；界面使用 42 |

主要结果字段：

- `MinimaxEscapeResult`：`agent_path`、`demon_path`、`success`、`demon_start`、`caught_step`、`score`。
- `ValueIterationResult`：`values`、`policy`、`path`、`iterations`、`converged`。
- `PolicyIterationResult`：`values`、`policy`、`path`、`policy_iterations`、`evaluation_iterations`。
- `QLearningResult`：`q_values`、`policy`、`path`、`episode_rewards`、`episodes`。

`policy` 将坐标映射为动作字符串，终止状态映射为 `None`；`q_values` 的键是 `(坐标, 动作)`。返回对象保存在内存中，界面没有导出模型或轨迹文件的功能。

## 当前限制与常见问题

- **找不到 `map.json`：**确认工作目录是仓库根目录。界面不会根据脚本所在目录自动修正地图路径。
- **提示缺少 Pygame：**使用启动程序的同一个解释器执行 `python -m pip install pygame`。
- **图片缺失或加载失败：**程序会退回色块/简单形状；实际加载的是 `plain.png` 和 `wall.png`，不是旧素材说明中的 `floor.png` 等名称。
- **点击开始后暂时无响应：**算法在界面线程同步运行。全金币 DP 和深度 15 的 Expectimax 可能需要等待；动画速度按钮只影响播放，不减少计算量。
- **切换/重启后没有动画：**新数据在开始时准备；按上文先暂停、选择模式/重启、再开始的顺序操作。`Restart` 还会清除搜索算法选择，必要时重新选择。
- **显示 `Caught` 但没有发生碰撞：**界面将所有 `success=False` 简化为 `Caught`，其中可能包括步数耗尽。代码实验应同时检查 `caught_step` 和最终位置。
- **收集或策略没有达到预期：**各模式目标不同；尤其 MDP/RL 的金币可重复给奖，展示路径还有防循环启发式，不能直接与全金币收集任务比较最优性。
- **自定义地图或参数的边界情况：**地图校验、不可达路径、零步数模拟和迭代参数检查尚不完善。使用正的回合/步数上限和 `0 <= gamma < 1`；策略迭代的内部评估循环没有单独的迭代次数上限。
- **运行中修改墙布局：**BFS 距离缓存按网格对象标识和目标保存，修改同一对象的墙不会自动使缓存失效；重新启动进程可清除缓存。

本次文档验证运行了三个搜索生成器、金币收集 DP、值迭代、策略迭代、固定种子的 Q-Learning，以及浅深度的 Minimax/Expectimax；检查了示例轨迹的合法性和默认地图结果。验证环境未安装 Pygame，因此没有实测桌面窗口交互，界面说明依据事件处理与绘制代码核对。
