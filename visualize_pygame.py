import sys, random, pygame
from typing import List, Tuple, Optional, Dict, Set
from grid import GridWorld
from algorithms import (
    best_collect_all_golds_dp,
    dfs_search_steps, greedy_search_steps, astar_search_steps, SearchStep,
    minimax_escape, minimax_escape_expectimax, MinimaxEscapeResult,
    value_iteration_plan, policy_iteration_plan, q_learning_plan,
    ValueIterationResult, PolicyIterationResult, QLearningResult,
)

# ---- Layout & Colors ----
CELL = 54
MARGIN = 2
PADDING = 40
MENU_H = 36      # 顶部菜单栏高度
DROPDOWN_W = 220 # 下拉面板宽度
ITEM_H = 28      # 下拉项高度
GAP = 8

COL_BG   = (245, 246, 248)
COL_BAR  = (252, 252, 252)
COL_MENU = (238, 241, 245)
COL_MENU_SEP = (210,210,210)
COL_TEXT = (40, 40, 40)
COL_ITEM = (250,250,250)
COL_ITEM_HOVER = (220,235,255)
COL_TOGGLE_ON  = (220,255,220)

COL_WALL = (90, 90, 90)
COL_TRAP = (220, 70, 70)
COL_GOLD = (240, 200, 60)
COL_PLAIN= (230, 233, 237)
COL_START= (60, 170, 90)
COL_GOAL = (255, 170, 0)

COL_DFS    = (70, 130, 180)
COL_GREEDY = (180, 70, 160)
COL_ASTAR  = (60, 170, 80)
COL_PART2  = (30, 144, 255)
COL_SHADOW = (190, 190, 190)
COL_FRONTIER = (255, 140, 0)
COL_VISITED  = (120, 180, 255)
COL_CURRENT  = (220, 60, 60)
COL_VALUE_ITER = (100, 200, 100)
COL_POLICY_ITER = (200, 100, 200)
COL_QLEARNING = (200, 150, 50)

globals_tiles: Dict[str, Optional[pygame.Surface]] = {}

def cell_center(c, r):
    x = PADDING + MARGIN + c*(CELL+MARGIN) + CELL//2
    y = MENU_H + PADDING + MARGIN + r*(CELL+MARGIN) + CELL//2
    return x, y

def load_tile_images(cell_size):
    names = {
        'S': 'start.png','G': 'goal.png','B': 'trap.png','$': 'gold.png',
        '#': 'wall.png','P': 'plain.png','agent': 'agent.png','demon': 'demon.png',
    }
    imgs: Dict[str, Optional[pygame.Surface]] = {}
    # base = __file__.rsplit('/',1)[0]
    import os
    for key, fname in names.items():
        path = os.path.join("assets", fname)
        try:
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                imgs[key] = pygame.transform.smoothscale(img, (cell_size, cell_size))
            else:
                imgs[key] = None
        except Exception:
            imgs[key] = None
    return imgs

def draw_grid(screen, gw, tiles):
    screen.fill(COL_BG)
    pygame.draw.rect(screen, COL_BAR, pygame.Rect(0,0,screen.get_width(), MENU_H))
    pygame.draw.line(screen, COL_MENU_SEP, (0, MENU_H-1), (screen.get_width(), MENU_H-1))
    for r in range(gw.h):
        for c in range(gw.w):
            x = PADDING + MARGIN + c*(CELL+MARGIN)
            y = MENU_H + PADDING + MARGIN + r*(CELL+MARGIN)
            t = gw.cell_type((c,r))
            symbol = t if t in ('S','G','B','$','#') else 'P'
            tile = tiles.get(symbol)
            if tile is not None:
                screen.blit(tile, (x, y))
            else:
                rect = pygame.Rect(x, y, CELL, CELL)
                if symbol == '#': color = COL_WALL
                elif symbol == 'B': color = COL_TRAP
                elif symbol == '$': color = COL_GOLD
                elif symbol == 'S': color = COL_START
                elif symbol == 'G': color = COL_GOAL
                else: color = COL_PLAIN
                pygame.draw.rect(screen, color, rect, border_radius=6)
    if gw.demon is not None:
        dc, dr = gw.demon
        x = PADDING + MARGIN + dc*(CELL+MARGIN)
        y = MENU_H + PADDING + MARGIN + dr*(CELL+MARGIN)
        tile = tiles.get('demon')
        if tile is not None:
            screen.blit(tile, (x, y))
        else:
            rect = pygame.Rect(x + CELL//4, y + CELL//4, CELL//2, CELL//2)
            pygame.draw.rect(screen, (200, 60, 60), rect, border_radius=6)

def draw_polyline(screen, pts, color, width=4):
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, width)

# ---- Dropdown UI ----
class MenuItem:
    def __init__(self, label, action=None, toggle=False, get_state=None):
        self.label = label
        self.action = action          # callable()
        self.toggle = toggle          # 是否是开关项
        self.get_state = get_state    # callable()->bool，返回开关当前是否开启
        self.hover = False

class Dropdown:
    def __init__(self, rect, title, items):
        self.rect = rect
        self.title = title
        self.items = items
        self.open = False

    def draw(self, screen, font):
        # 顶部菜单按钮
        bg = COL_MENU if not self.open else (225,232,245)
        pygame.draw.rect(screen, bg, self.rect, border_radius=6)
        pygame.draw.rect(screen, COL_MENU_SEP, self.rect, width=1, border_radius=6)
        label = font.render(self.title, True, COL_TEXT)
        screen.blit(label, (self.rect.x+10, self.rect.y + (self.rect.h-label.get_height())//2))
        # 下拉面板
        if self.open:
            panel = pygame.Rect(self.rect.x, self.rect.bottom+2, DROPDOWN_W, ITEM_H*len(self.items)+6)
            pygame.draw.rect(screen, (255,255,255), panel, border_radius=8)
            pygame.draw.rect(screen, COL_MENU_SEP, panel, width=1, border_radius=8)
            # 每一项
            for i, it in enumerate(self.items):
                r = pygame.Rect(panel.x+3, panel.y+3 + i*ITEM_H, panel.w-6, ITEM_H-4)
                it.hover = r.collidepoint(pygame.mouse.get_pos())
                bg = COL_ITEM_HOVER if it.hover else COL_ITEM
                if it.toggle and it.get_state and it.get_state():
                    bg = COL_TOGGLE_ON
                pygame.draw.rect(screen, bg, r, border_radius=6)
                txt = font.render(it.label, True, COL_TEXT)
                screen.blit(txt, (r.x+10, r.y+(r.h-txt.get_height())//2))

    def handle_click(self, pos):
        if self.rect.collidepoint(pos):
            self.open = not self.open
            return True, None
        if self.open:
            # 点击面板中的条目
            panel_top = self.rect.bottom+2
            panel = pygame.Rect(self.rect.x, panel_top, DROPDOWN_W, ITEM_H*len(self.items)+6)
            if panel.collidepoint(pos):
                idx = (pos[1] - (panel_top+3)) // ITEM_H
                if 0 <= idx < len(self.items):
                    self.open = False
                    return True, self.items[idx]
            else:
                # 点击其它区域关闭
                self.open = False
        return False, None

# ---- Animations ----
class PathAnim:
    def __init__(self, path, color=(0,0,0), width=5, step_ms=220):
        self.path = path or []
        self.color = color
        self.width = width
        self.step_ms = step_ms
        self.idx = 0
        self.timer = 0
        self.play = False  # 初始为暂停，只有点击“开始/暂停”按钮才开始
        self.centers = [cell_center(c,r) for (c,r) in self.path]
        self.visible = True

    def reset(self):
        self.idx = 0
        self.timer = 0
        # 不改变 self.play，避免重置时自动播放

    def update(self, dt):
        if not self.play or len(self.path) < 2: return
        self.timer += dt
        while self.timer >= self.step_ms and self.idx < len(self.path)-1:
            self.timer -= self.step_ms; self.idx += 1
    def draw(self, screen):
        if not self.visible: return
        if len(self.centers) >= 2:
            upto = max(2, self.idx+1)
            draw_polyline(screen, self.centers[:upto], self.color, self.width)
        if self.centers:
            head = self.centers[self.idx]
            pygame.draw.circle(screen, self.color, head, 8)

def reconstruct_to(parent, cur):
    path = []
    t = cur
    while t is not None and t in parent:
        path.append(t); t = parent[t]
    return list(reversed(path))

class SearchAnimator:
    def __init__(self, steps, step_ms=120):
        self.steps = steps; self.idx = 0; self.timer = 0
        self.step_ms = step_ms; self.play = False  # 初始暂停
        self.algo = steps[0].algo if steps else ""
    def reset(self):
        self.idx = 0
        self.timer = 0

    def update(self, dt):
        if not self.play or not self.steps: return
        self.timer += dt
        while self.timer >= self.step_ms and self.idx < len(self.steps)-1:
            self.timer -= self.step_ms; self.idx += 1
    def draw(self, screen):
        if not self.steps: return
        st = self.steps[self.idx]
        for v in st.visited:
            x = PADDING + MARGIN + v[0]*(CELL+MARGIN)
            y = MENU_H + PADDING + MARGIN + v[1]*(CELL+MARGIN)
            pygame.draw.rect(screen, COL_VISITED, (x,y,CELL,CELL), border_radius=6)
        for f in st.frontier:
            x = PADDING + MARGIN + f[0]*(CELL+MARGIN)
            y = MENU_H + PADDING + MARGIN + f[1]*(CELL+MARGIN)
            pygame.draw.rect(screen, COL_FRONTIER, (x,y,CELL,CELL), border_radius=6)
        if st.current is not None:
            x = PADDING + MARGIN + st.current[0]*(CELL+MARGIN)
            y = MENU_H + PADDING + MARGIN + st.current[1]*(CELL+MARGIN)
            pygame.draw.rect(screen, COL_CURRENT, (x,y,CELL,CELL), border_radius=6)
            pts = [cell_center(c,r) for (c,r) in reconstruct_to(st.parent, st.current)]
            draw_polyline(screen, pts, (0,0,0), 3)

class AgentPathAnim:
    def __init__(self, gw, path, step_ms=160):
        self.gw = gw
        self.path = path or []
        self.centers = [cell_center(c, r) for (c, r) in self.path]
        self.step_ms = step_ms
        self.idx = 0
        self.timer = 0
        self.play = False  # 初始暂停，仅按钮控制
        # 记录初始的金矿位置与已采集集合
        self.orig_golds: Set[Tuple[int,int]] = {pos for pos, t in self.gw.cells.items() if t == '$'}
        self.collected: Set[Tuple[int,int]] = set()

    def reset(self):
        # 重置并恢复已采集的金矿
        for pos in self.collected:
            self.gw.cells[pos] = '$'
        self.collected.clear()
        self.idx = 0
        self.timer = 0
        # 不改 play

    def update(self, dt):
        if not self.play or len(self.path) < 2: return
        self.timer += dt
        while self.timer >= self.step_ms and self.idx < len(self.path) - 1:
            self.timer -= self.step_ms
            prev = self.path[self.idx]   # 将要离开的格子
            self.idx += 1                # 移动到下一格
            # 若刚离开的格子原本是金矿，则此时视为采集完成 -> 变为道路
            if prev in self.orig_golds and prev not in self.collected:
                self.gw.cells[prev] = 'P'
                self.collected.add(prev)

    def draw(self, screen):
        if not self.centers: return
        cx, cy = self.centers[self.idx]
        img = globals_tiles.get('agent')
        if img is not None:
            rect = img.get_rect(center=(cx, cy))
            screen.blit(img, rect.topleft)
        else:
            s = CELL // 2
            pygame.draw.rect(screen, (60,170,90), (cx - s//2, cy - s//2, s, s), border_radius=6)

# ---- Minimax escape animation ----
class MinimaxAnimator:
    def __init__(self, gw, result, step_ms=200):
        self.gw = gw
        self.step_ms = step_ms
        self.play = False
        self.result: Optional[MinimaxEscapeResult] = None
        self.agent_centers: List[Tuple[int,int]] = []
        self.demon_centers: List[Tuple[int,int]] = []
        self.idx = 0
        self.timer = 0
        self.set_result(result)

    def set_result(self, result):
        self.result = result
        self.agent_centers = [cell_center(c, r) for (c, r) in (result.agent_path if result else [])]
        self.demon_centers = [cell_center(c, r) for (c, r) in (result.demon_path if result else [])]
        self.idx = 0
        self.timer = 0
        self._apply_positions()

    def _apply_positions(self):
        if not self.result:
            self.gw.demon = None
            return
        if self.result.demon_path:
            demon_idx = min(self.idx, len(self.result.demon_path) - 1)
            self.gw.demon = self.result.demon_path[demon_idx]
        elif self.result.demon_start is not None:
            self.gw.demon = self.result.demon_start
        else:
            self.gw.demon = None

    def reset(self):
        self.idx = 0
        self.timer = 0
        self._apply_positions()

    def update(self, dt):
        if not self.play or not self.result or len(self.agent_centers) < 2:
            self._apply_positions()
            return
        self.timer += dt
        advanced = False
        while self.timer >= self.step_ms and self.idx < len(self.agent_centers) - 1:
            self.timer -= self.step_ms
            self.idx += 1
            advanced = True
        if advanced:
            self._apply_positions()

    def draw(self, screen):
        if not self.result or not self.agent_centers:
            return
        self._apply_positions()
        upto = max(1, self.idx + 1)
        draw_polyline(screen, self.agent_centers[:upto], COL_PART2, 4)
        ax, ay = self.agent_centers[self.idx]
        agent_img = globals_tiles.get('agent')
        if agent_img is not None:
            rect = agent_img.get_rect(center=(ax, ay))
            screen.blit(agent_img, rect.topleft)
        else:
            pygame.draw.circle(screen, COL_START, (ax, ay), CELL//3)
        if self.demon_centers:
            upto_d = min(self.idx + 1, len(self.demon_centers))
            if upto_d >= 2:
                draw_polyline(screen, self.demon_centers[:upto_d], (200, 60, 60), 3)

# ---- RL/DP Policy Animation ----
class PolicyPathAnim:
    """动画器：用于值迭代、策略迭代、Q学习等基于策略的路径可视化"""
    def __init__(self, gw, result, color=(100, 200, 100), step_ms=160):
        self.gw = gw
        self.result = result
        self.path = result.path if result else []
        self.centers = [cell_center(c, r) for (c, r) in self.path]
        self.color = color
        self.step_ms = step_ms
        self.idx = 0
        self.timer = 0
        self.play = False
        
    def set_result(self, result, color=None):
        """更新结果和路径"""
        self.result = result
        self.path = result.path if result else []
        self.centers = [cell_center(c, r) for (c, r) in self.path]
        if color is not None:
            self.color = color
        self.idx = 0
        self.timer = 0
        
    def reset(self):
        self.idx = 0
        self.timer = 0
        
    def update(self, dt):
        if not self.play or len(self.path) < 2:
            return
        self.timer += dt
        while self.timer >= self.step_ms and self.idx < len(self.path) - 1:
            self.timer -= self.step_ms
            self.idx += 1
            
    def draw(self, screen):
        if not self.centers:
            return
        # 绘制已经走过的路径
        if len(self.centers) >= 2:
            upto = max(2, self.idx + 1)
            draw_polyline(screen, self.centers[:upto], self.color, 4)
        # 绘制当前位置的智能体
        cx, cy = self.centers[self.idx]
        agent_img = globals_tiles.get('agent')
        if agent_img is not None:
            rect = agent_img.get_rect(center=(cx, cy))
            screen.blit(agent_img, rect.topleft)
        else:
            pygame.draw.circle(screen, self.color, (cx, cy), CELL//3)

# ---- App ----
def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python visualize_pygame.py map.json"); sys.exit(1)
    gw = GridWorld.from_json()

    pygame.init()
    font = pygame.font.SysFont("Arial", 18)
    W = gw.w*CELL + (gw.w+1)*MARGIN + 2*PADDING
    H = MENU_H + gw.h*CELL + (gw.h+1)*MARGIN + 2*PADDING + 40
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GridWorld (Dropdown Menu)")

    global globals_tiles
    globals_tiles = load_tile_images(CELL)

    # 预计算
    steps_dfs    = list(dfs_search_steps(gw, gw.start, gw.goal))
    steps_greedy = list(greedy_search_steps(gw, gw.start, gw.goal))
    steps_astar  = list(astar_search_steps(gw, gw.start, gw.goal))
    res2 = best_collect_all_golds_dp(gw)
    p_collect = res2.path
    
    # 预计算值迭代、策略迭代和Q学习
    value_iter_result = value_iteration_plan(gw, gamma=0.9, theta=1e-4, max_iterations=1000)
    policy_iter_result = policy_iteration_plan(gw, gamma=0.9, theta=1e-4, max_policy_iterations=100)
    qlearning_result = q_learning_plan(gw, episodes=500, gamma=0.9, alpha=0.1, epsilon=0.1, seed=42)

    # 动画器：Collect-All 使用 AgentPathAnim
    anim_part2  = AgentPathAnim(gw, p_collect, step_ms=160)
    anim_search = SearchAnimator(steps_dfs, step_ms=120)
    anim_minimax = MinimaxAnimator(gw, None, step_ms=200)
    
    # 动画器：值迭代、策略迭代、Q学习
    anim_value_iter = PolicyPathAnim(gw, value_iter_result, color=COL_VALUE_ITER, step_ms=160)
    anim_policy_iter = PolicyPathAnim(gw, policy_iter_result, color=COL_POLICY_ITER, step_ms=160)
    anim_qlearning = PolicyPathAnim(gw, qlearning_result, color=COL_QLEARNING, step_ms=160)

    rng = random.Random()
    minimax_result: Optional[MinimaxEscapeResult] = None

    def recompute_minimax():
        nonlocal minimax_result
        gw.randomize_demon(rng)
        seed = rng.randrange(10**9)
        # 使用 Expectimax 算法（更适合随机移动的魔鬼）
        minimax_result = minimax_escape_expectimax(gw, depth=15, max_steps=1000, demon_start=gw.demon, seed=seed)
        anim_minimax.set_result(minimax_result)

    recompute_minimax()

    # 模式/播放
    MODE_SEARCH = 4
    MODE_COLLECT = 2
    MODE_MINIMAX = 6
    MODE_VALUE_ITER = 7
    MODE_POLICY_ITER = 8
    MODE_QLEARNING = 9
    MODE = MODE_SEARCH
    is_playing = False
    def apply_play_state():
        anim_part2.play = is_playing
        anim_search.play = is_playing
        anim_minimax.play = is_playing
        anim_value_iter.play = is_playing
        anim_policy_iter.play = is_playing
        anim_qlearning.play = is_playing
    def set_playing(state: bool):
        nonlocal is_playing
        is_playing = state
        apply_play_state()

    # ---- Dropdown menus ----
    menu_x = PADDING
    mode_rect     = pygame.Rect(menu_x, 4, 120, MENU_H-8); menu_x += mode_rect.w + GAP
    controls_rect = pygame.Rect(menu_x, 4, 120, MENU_H-8); menu_x += controls_rect.w + GAP
    algo_rect     = pygame.Rect(menu_x, 4, 140, MENU_H-8); menu_x += algo_rect.w + GAP
    pause_rect    = pygame.Rect(menu_x, 4, 120, MENU_H-8)

    # 移除 Overlay/Shadow 开关
    # def is_on(...) / do_toggle(...) 已删除

    # 构造菜单项（删除 Overlay）
    mode_items = [
        MenuItem("Part1: Search",       lambda: set_mode(MODE_SEARCH)),
        MenuItem("Part2: Collect-All",  lambda: set_mode(MODE_COLLECT)),
        MenuItem("Part3: Minimax",      lambda: set_mode(MODE_MINIMAX)),
        MenuItem("Part4: Value Iter",   lambda: set_mode(MODE_VALUE_ITER)),
        MenuItem("Part5: Policy Iter",  lambda: set_mode(MODE_POLICY_ITER)),
        MenuItem("Part6: Q-Learning",   lambda: set_mode(MODE_QLEARNING)),
    ]
    control_items = [
        MenuItem("Restart",    lambda: on_restart()),
        MenuItem("Speed -",    lambda: on_speed(-1)),
        MenuItem("Speed +",    lambda: on_speed(+1)),
    ]
    algo_items = [
        MenuItem("DFS",           lambda: set_algo("dfs")),
        MenuItem("Greedy",        lambda: set_algo("greedy")),
        MenuItem("A*",            lambda: set_algo("astar")),
    ]

    dropdowns = [
        Dropdown(mode_rect, "Mode", mode_items),
        Dropdown(controls_rect, "Controls", control_items),
        Dropdown(algo_rect, "Algorithm", algo_items),
    ]

    # 按钮（开始/暂停）
    class Button:
        def __init__(self, rect, label, on_click, get_state=None):
            self.rect = rect; self.label = label; self.on_click = on_click; self.get_state = get_state
        def draw(self, screen, font):
            active = self.get_state() if self.get_state else False
            bg = (200, 235, 200) if active else COL_MENU
            pygame.draw.rect(screen, bg, self.rect, border_radius=6)
            pygame.draw.rect(screen, COL_MENU_SEP, self.rect, width=1, border_radius=6)
            txt = font.render(self.label, True, COL_TEXT)
            screen.blit(txt, (self.rect.x+10, self.rect.y + (self.rect.h-txt.get_height())//2))
        def handle_click(self, pos):
            if self.rect.collidepoint(pos):
                if self.on_click: self.on_click()
                return True
            return False

    play_button = Button(pause_rect, "Start/Pause", lambda: on_play_pause(), get_state=lambda: is_playing)

    # ---- 菜单回调 ----
    def set_mode(m):
        nonlocal MODE
        MODE = m
        if MODE == MODE_MINIMAX:
            anim_minimax.reset()
        elif MODE == MODE_VALUE_ITER:
            anim_value_iter.reset()
        elif MODE == MODE_POLICY_ITER:
            anim_policy_iter.reset()
        elif MODE == MODE_QLEARNING:
            anim_qlearning.reset()
        apply_play_state()

    def set_algo(name):
        nonlocal MODE
        if name == "dfs":
            MODE = MODE_SEARCH
            anim_search.steps = steps_dfs
            anim_search.algo = "DFS"
            anim_search.reset()
        elif name == "greedy":
            MODE = MODE_SEARCH
            anim_search.steps = steps_greedy
            anim_search.algo = "Greedy"
            anim_search.reset()
        elif name == "astar":
            MODE = MODE_SEARCH
            anim_search.steps = steps_astar
            anim_search.algo = "A*"
            anim_search.reset()
        elif name == "collect":
            MODE = MODE_COLLECT
            anim_part2.reset()
        elif name == "minimax":
            MODE = MODE_MINIMAX
            recompute_minimax()
            anim_minimax.reset()
        elif name == "value_iter":
            MODE = MODE_VALUE_ITER
            anim_value_iter.reset()
        elif name == "policy_iter":
            MODE = MODE_POLICY_ITER
            anim_policy_iter.reset()
        elif name == "qlearning":
            MODE = MODE_QLEARNING
            anim_qlearning.reset()
        apply_play_state()

    def on_play_pause():
        set_playing(not is_playing)

    def on_restart():
        anim_part2.reset()
        anim_search.reset()
        recompute_minimax()
        anim_minimax.reset()
        anim_value_iter.reset()
        anim_policy_iter.reset()
        anim_qlearning.reset()
        apply_play_state()

    def on_speed(delta):
        step = -20 if delta>0 else +20
        anim_part2.step_ms = max(40, anim_part2.step_ms + step)
        anim_search.step_ms = max(40, anim_search.step_ms + step)
        anim_minimax.step_ms = max(60, anim_minimax.step_ms + step)
        anim_value_iter.step_ms = max(40, anim_value_iter.step_ms + step)
        anim_policy_iter.step_ms = max(40, anim_policy_iter.step_ms + step)
        anim_qlearning.step_ms = max(40, anim_qlearning.step_ms + step)

    # ---- Main loop ----
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                clicked_any = False
                for d in dropdowns:
                    handled, item = d.handle_click(e.pos)
                    if handled:
                        clicked_any = True
                        if item and item.action:
                            item.action()
                if not clicked_any and play_button.handle_click(e.pos):
                    clicked_any = True
                if not clicked_any:
                    for d in dropdowns: d.open = False

        # 更新动画（无 Overlay）
        if MODE == MODE_COLLECT:
            anim_part2.update(dt)
        elif MODE == MODE_MINIMAX:
            anim_minimax.update(dt)
        elif MODE == MODE_VALUE_ITER:
            anim_value_iter.update(dt)
        elif MODE == MODE_POLICY_ITER:
            anim_policy_iter.update(dt)
        elif MODE == MODE_QLEARNING:
            anim_qlearning.update(dt)
        else:
            anim_search.update(dt)

        # 绘制
        draw_grid(screen, gw, globals_tiles)
        if MODE == MODE_COLLECT:
            anim_part2.draw(screen)
        elif MODE == MODE_MINIMAX:
            anim_minimax.draw(screen)
        elif MODE == MODE_VALUE_ITER:
            anim_value_iter.draw(screen)
        elif MODE == MODE_POLICY_ITER:
            anim_policy_iter.draw(screen)
        elif MODE == MODE_QLEARNING:
            anim_qlearning.draw(screen)
        else:
            anim_search.draw(screen)

        for d in dropdowns: d.draw(screen, font)
        play_button.draw(screen, font)

        # 底部信息栏
        minimax_info = "N/A"
        if minimax_result:
            status = "Safe" if minimax_result.success else "Caught"
            minimax_info = f"{status} len={len(minimax_result.agent_path)}"
        
        # 构建信息字符串
        if MODE == MODE_VALUE_ITER:
            info = (
                f"Value Iteration: path_len={len(value_iter_result.path)} "
                f"iterations={value_iter_result.iterations} "
                f"converged={value_iter_result.converged}"
            )
        elif MODE == MODE_POLICY_ITER:
            info = (
                f"Policy Iteration: path_len={len(policy_iter_result.path)} "
                f"policy_iters={policy_iter_result.policy_iterations} "
                f"eval_iters={policy_iter_result.evaluation_iterations}"
            )
        elif MODE == MODE_QLEARNING:
            avg_reward = sum(qlearning_result.episode_rewards[-100:]) / min(100, len(qlearning_result.episode_rewards)) if qlearning_result.episode_rewards else 0
            info = (
                f"Q-Learning: path_len={len(qlearning_result.path)} "
                f"episodes={qlearning_result.episodes} "
                f"avg_reward(last100)={avg_reward:.1f}"
            )
        else:
            info = (
                f"Start {gw.start}  Goal {gw.goal} | Part2 length={len(p_collect) if p_collect else 'N/A'}"
                f" | Minimax {minimax_info}"
            )
        
        txt = font.render(info, True, (60,60,60))
        screen.blit(txt, (PADDING, H - txt.get_height() - 10))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
