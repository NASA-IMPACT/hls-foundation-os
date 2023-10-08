import collections
from typing import Union

import numpy as np
import rasterio


def find_av_path(
    grid: Union[str, np.array], start: tuple[int, int], end: tuple[int, int]
) -> np.ndarray:
    if isinstance(grid, str):
        grid = load_img(grid)

    height, width = grid.shape
    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == end:
            return get_mask_from_path(path, grid.shape)
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                0 <= x2 < width
                and 0 <= y2 < height
                and grid[y2][x2] != 1
                and (x2, y2) not in seen
            ):
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def load_img(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def get_mask_from_path(path: list[tuple[int, int]], shape: tuple) -> np.ndarray:
    mask = np.zeros(shape)
    for position in path:
        mask[position] = 1
    return mask
