from typing import List, Tuple, Dict, Set, Optional
import json, os, random

Coord = Tuple[int, int]  # (col, row)

TILE_PLAIN = 'P'
TILE_WALL  = '#'
TILE_TRAP  = 'B'
TILE_GOLD  = '$'
TILE_GOAL  = 'G'
TILE_START = 'S'

class GridWorld:
    def __init__(self, width = 10, height = 10):
        self.w = width
        self.h = height
        self.cells: Dict[Coord, str] = {}
        self.start: Coord = (0, 0)
        self.goal: Coord = (0, 9)
        self.step_cost = -1
        self.rewards = {'G': 20, '$': 10, 'B': -10}
        self._gold_taken: Set[Coord] = set()
        self.demon: Optional[Coord] = None

    @staticmethod
    def from_json(path = 'map.json'):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layout: List[List[str]] = data["layout"]
        h = len(layout); w = len(layout[0]) if h>0 else 0
        g = GridWorld(w, h)
        for r in range(h):
            for c in range(w):
                t = layout[r][c]
                if t == 'S':
                    g.start = (c, r)
                elif t == 'G':
                    g.goal = (c, r)
                elif t == '#':
                    g.cells[(c,r)] = TILE_WALL
                elif t == 'B':
                    g.cells[(c,r)] = TILE_TRAP
                elif t == '$':
                    g.cells[(c,r)] = TILE_GOLD
                else:
                    # 'P' treated as plain corridor

                    pass
        return g

    def in_bounds(self, p):
        c, r = p
        return 0 <= c < self.w and 0 <= r < self.h

    def is_wall(self, p):
        return self.cells.get(p, TILE_PLAIN) == TILE_WALL

    def cell_type(self, p):
        if p == self.start: return TILE_START
        if p == self.goal:  return TILE_GOAL
        return self.cells.get(p, TILE_PLAIN)

    # --- Reward helpers -------------------------------------------------
    def reward_for_type(self, t):
        if t == TILE_GOAL:
            return 20
        if t == TILE_TRAP:
            return -10
        if t == TILE_GOLD:
            return 10
        if t in (TILE_START, TILE_PLAIN):
            return -1
        if t == TILE_WALL:
            return -5
        return -1

    def reward_for(self, p):
        """Return the instantaneous reward for moving into ``p``."""
        return self.reward_for_type(self.cell_type(p))

    @property
    def wall_penalty(self):
        return self.reward_for_type(TILE_WALL)

    def is_terminal(self, p):
        return self.cell_type(p) in (TILE_GOAL, TILE_TRAP)

    def neighbors4(self, p):
        c, r = p
        res = [(c+1, r), (c-1, r), (c, r+1), (c, r-1)]
        return [q for q in res if self.in_bounds(q) and not self.is_wall(q)]

    def randomize_demon(self, rng = None):
        """Place the demon on a random traversable tile (excluding start/goal)."""
        rng = rng or random
        candidates: List[Coord] = []
        for r in range(self.h):
            for c in range(self.w):
                coord = (c, r)
                if coord == self.start or coord == self.goal:
                    continue
                if self.is_wall(coord):
                    continue
                candidates.append(coord)
        if not candidates:
            self.demon = None
        else:
            self.demon = rng.choice(candidates)
        return self.demon
