Place your 8 icons into this folder with the exact filenames (PNG recommended):
  - start.png     # S  入口 (-1)
  - goal.png      # G  出口 (+20)
  - trap.png      # B  陷阱 (-10)
  - gold.png      # $  金矿 (+10)
  - obstacle.png  # O  障碍 (-5, 可通行，附加代价)
  - floor.png     # .  通道 (-1)
  - agent.png     # 智能体
  - demon.png     # 魔鬼

Optional (if you also want hard walls):
  - wall.png      # W  墙（不可通行）

Images will be auto-scaled to the tile size (CELL) in visualize_pygame.py.
If a file is missing, the program falls back to color blocks.
