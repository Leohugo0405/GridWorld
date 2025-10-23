
from PIL import Image
from typing import Dict, Tuple
import os

def slice_legend(atlas_path: str, cell_px: int = 128) -> Dict[str, Image.Image]:
    """
    Assume the legend image is arranged roughly as:
      Row 0 (top):  S | G | B | Gold | Wall
      Row 1 (bottom): Corridor | Agent | (empty) | (empty) | Demon
    The function crops by equal columns and halves vertically.
    Returns PIL Images keyed by logical names.
    """
    im = Image.open(atlas_path).convert("RGBA")
    W, H = im.size
    # Split rows by half (with a small margin to avoid text below)
    split_y = H//2

    col_w = W // 5
    def crop(col, row):
        x0 = col * col_w
        x1 = (col+1) * col_w
        if row == 0:
            y0, y1 = 0, split_y
        else:
            y0, y1 = split_y, H
        # Trim a small margin
        pad = int(0.06 * col_w)
        ypad = int(0.06 * (y1-y0))
        x0 += pad; x1 -= pad; y0 += ypad; y1 -= ypad
        tile = im.crop((x0, y0, x1, y1))
        # square fit
        s = max(tile.size)
        canvas = Image.new("RGBA", (s, s), (0,0,0,0))
        canvas.paste(tile, ((s - tile.size[0])//2, (s - tile.size[1])//2), tile)
        return canvas

    tiles = {
        "S": crop(0,0),
        "G": crop(1,0),
        "B": crop(2,0),
        "$": crop(3,0),
        "#": crop(4,0),     # wall (grey)
        "P": crop(0,1),     # corridor (white)
        "agent": crop(1,1), # pacman
        "demon": crop(4,1), # demon
    }
    # Resize to target cell if specified
    if cell_px:
        for k, v in tiles.items():
            tiles[k] = v.resize((cell_px, cell_px), Image.LANCZOS)
    return tiles
