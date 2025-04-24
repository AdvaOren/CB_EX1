#!/usr/bin/env python3
"""
Block Automaton Simulation with Pygame GUI and Block Partition Overlay

This script implements a 2D block-based cellular automaton with two alternating environments:
- Odd generations: blocks partitioned starting at (0,0), stepping by 2 (blue lines).
- Even generations: blocks partitioned starting at (1,1), stepping by 2 (red dashed lines), with optional wrap-around.

Rules per 2x2 block:
- If exactly 2 cells are alive: no change.
- If 0, 1, or 4 cells are alive: invert all cells (0→1, 1→0).
- If exactly 3 cells are alive: invert all cells, then rotate the block 180°.

Metrics:
- Density: fraction of living cells each generation.
- Stability: fraction of cells unchanged from the previous generation.

Usage:
    python block_automaton.py --size 100 [--prob 0.5] [--wrap] [--steps 250] [--cell-size 5] [--delay 200]

Dependencies:
    pip install pygame numpy matplotlib

Close the window or press CTRL+C in the console to exit early.
"""
import sys
import argparse
import time

import numpy as np
import pygame

# Global display settings
default_cell_size = 5
MARGIN = 40
MAX_WINDOW_DIM = 600  # target maximum window height (incl. margin)


def parse_args():
    parser = argparse.ArgumentParser(description="Block Automaton Simulation with Pygame GUI")
    parser.add_argument(
        '--size', type=int, default=100,
        help='Grid size (NxN), even number (default: 100)'
    )
    parser.add_argument(
        '--prob', type=float, default=0.5,
        help='Initial probability for state 1 (default: 0.5)'
    )
    parser.add_argument(
        '--wrap', action='store_true',
        help='Enable wrap-around blocks (toroidal grid)'
    )
    parser.add_argument(
        '--steps', type=int, default=250,
        help='Number of generations to simulate (default: 250)'
    )
    parser.add_argument(
        '--cell-size', type=int, default=None,
        help='Size of each cell in pixels (default: auto or 5)'
    )
    parser.add_argument(
        '--delay', type=int, default=0,
        help='Delay between generations in milliseconds (default: 0)'
    )
    return parser.parse_args()


def initialize_grid(N, prob):
    return np.random.choice([0, 1], size=(N, N), p=[1 - prob, prob])


def get_block_positions(N, gen, wrap):
    if gen % 2 == 1:
        starts = list(range(0, N, 2))
    else:
        if wrap:
            starts = [(1 + 2 * k) % N for k in range((N + 1) // 2)]
        else:
            starts = list(range(1, N - 1, 2))
    return [(i, j) for i in starts for j in starts]


def update_grid(grid, gen, wrap):
    N = grid.shape[0]
    new_grid = grid.copy()
    for i, j in get_block_positions(N, gen, wrap):
        coords = [((i + di) % N, (j + dj) % N) for di in (0, 1) for dj in (0, 1)]
        block = np.array([grid[x, y] for x, y in coords]).reshape(2, 2)
        s = block.sum()
        if s == 2:
            continue
        new_block = 1 - block
        if s == 3:
            new_block = np.rot90(new_block, 2)
        for idx, (x, y) in enumerate(coords):
            new_grid[x, y] = new_block[idx // 2, idx % 2]
    return new_grid


def draw_block_lines(screen, gen, cell_size, margin, N, wrap):
    dash, gap = 5, 5
    height = N * cell_size
    width = N * cell_size
    if gen % 2 == 1:
        color, offsets, dashed = (0, 0, 255), list(range(0, N, 2)), False
    else:
        color, dashed = (255, 0, 0), True
        offsets = list(range(1, N, 2))
    for off in offsets:
        x = off * cell_size
        y0, y1 = margin, margin + height
        if not dashed:
            pygame.draw.line(screen, color, (x, y0), (x, y1), 2)
        else:
            y = y0
            while y < y1:
                y2 = min(y + dash, y1)
                pygame.draw.line(screen, color, (x, y), (x, y2), 1)
                y += dash + gap
        y = margin + off * cell_size
        x0, x1 = 0, width
        if not dashed:
            pygame.draw.line(screen, color, (x0, y), (x1, y), 2)
        else:
            x = x0
            while x < x1:
                x2 = min(x + dash, x1)
                pygame.draw.line(screen, color, (x, y), (x2, y), 1)
                x += dash + gap


def draw_grid(screen, grid, cell_size, margin):
    N = grid.shape[0]
    for i in range(N):
        for j in range(N):
            # now: dead cells = white, live cells = black
            c = (0, 0, 0) if grid[i, j] == 1 else (255, 255, 255)
            r = pygame.Rect(j * cell_size, i * cell_size + margin,
                             cell_size, cell_size)
            screen.fill(c, r)


def run_simulation(N, prob, wrap, steps, cell_size, delay):
    pygame.init()
    if cell_size is None:
        usable = MAX_WINDOW_DIM - MARGIN
        cell_size = max(default_cell_size, usable // N)
    width = N * cell_size
    height = N * cell_size + MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Block Automaton Simulation")
    font = pygame.font.SysFont(None, 24)

    grid = initialize_grid(N, prob)
    prev = None
    densities, stabilities = [], []
    clock = pygame.time.Clock()
    gen = 1

    while gen <= steps:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
        d = grid.mean(); s = 0.0 if prev is None else (grid == prev).mean()
        densities.append(d); stabilities.append(s)
        screen.fill((200, 200, 200), (0, 0, width, MARGIN))
        draw_grid(screen, grid, cell_size, MARGIN)
        draw_block_lines(screen, gen, cell_size, MARGIN, N, wrap)
        txt = f"Gen:{gen}/{steps}  Density:{d:.3f}  Stability:{s:.3f}"
        img = font.render(txt, True, (0, 0, 0)); screen.blit(img, (5, 5))
        pygame.display.flip()
        if delay > 0:
            pygame.time.delay(delay)
        clock.tick(60)
        prev = grid.copy(); grid = update_grid(grid, gen, wrap); gen += 1
    pygame.quit()
    try:
        import matplotlib.pyplot as plt
        plt.plot(range(1, steps+1), densities, label='Density')
        plt.plot(range(1, steps+1), stabilities, label='Stability')
        plt.xlabel('Gen'); plt.ylabel('Value'); plt.legend(); plt.show()
    except ImportError:
        with open('metrics.csv','w',newline='') as f:
            import csv; w=csv.writer(f); w.writerow(['Gen','Density','Stability'])
            w.writerows(enumerate(zip(densities,stabilities),1))
        print("Saved metrics.csv")


def main():
    args = parse_args()
    if args.size % 2 != 0:
        print("--size must be even"); sys.exit(1)
    run_simulation(
        args.size, args.prob, args.wrap, args.steps,
        args.cell_size, args.delay
    )


if __name__ == '__main__':
    main()