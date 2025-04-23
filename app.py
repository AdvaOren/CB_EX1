"""
Block Cellular Automaton Simulation with Pygame GUI

Assignment:
Implement a block-based cellular automaton on an N×N grid (N even). In odd generations, partition the grid into disjoint 2×2 blocks starting at (0,0), (0,2), ... (blue blocks); in even generations, use blocks offset by (1,1) (red blocks). Two modes:
  - No wraparound: ignore blocks that would cross the boundary.
  - Wraparound: use toroidal boundary conditions so all blocks participate.

Rules for each 2×2 block (independently):
 1. Count live cells (value 1) in the block.
 2. If count == 2: leave the block unchanged.
 3. If count in {0,1,4}: flip every cell in the block (0→1, 1→0).
 4. If count == 3: flip every cell, then rotate the block 180°.

GUI (pygame): display live cells as white squares, dead cells as black squares. Update in real time.

Usage:
    python block_automaton.py [--size N] [--prob P] [--wrap]

Options:
    --size N   Grid dimension N (even, default 100)
    --prob P   Initial live-cell probability (0.0–1.0, default 0.5)
    --wrap     Enable wraparound boundary conditions (default: off)

Requirements:
  - Python 3.x
  - pygame (install via `pip install pygame`)

Controls:
  - Close window or press ESC to quit.

"""
import pygame
import sys
import random
import argparse


def init_grid(N, prob):
    """Initialize an N×N grid with live (1) or dead (0) cells at random probability."""
    return [[1 if random.random() < prob else 0 for _ in range(N)] for _ in range(N)]


def get_block_positions(N, offset_x, offset_y, wrap):
    """
    Generate a list of all 2×2 block coordinate lists given an offset.
    If wrap=False, only include blocks fully inside the grid.
    If wrap=True, use toroidal wrapping via modulo arithmetic.
    """
    blocks = []
    i = offset_x
    while i < N + offset_x:
        j = offset_y
        while j < N + offset_y:
            if wrap:
                # Wrap around edges
                coords = [
                    (i % N,      j % N),
                    (i % N,      (j+1) % N),
                    ((i+1) % N,  j % N),
                    ((i+1) % N,  (j+1) % N)
                ]
                blocks.append(coords)
            else:
                # Only include fully inside blocks
                if i+1 < N and j+1 < N:
                    coords = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                    blocks.append(coords)
            j += 2
        i += 2
    return blocks


def apply_rules(grid, N, pattern, wrap):
    """
    Apply automaton rules to the entire grid for one generation.
    pattern: 'blue' or 'red' to choose block tiling.
    Returns a new grid.
    """
    # Determine offset based on pattern
    offset_x, offset_y = (0, 0) if pattern == 'blue' else (1, 1)
    blocks = get_block_positions(N, offset_x, offset_y, wrap)
    new_grid = [row[:] for row in grid]

    for block in blocks:
        cells = [grid[x][y] for x, y in block]
        s = sum(cells)
        if s == 2:
            # No change
            continue
        # Flip all cells
        flipped = [1 - c for c in cells]
        if s == 3:
            # Rotate flipped block 180°: reorder [0→3,1→2,2→1,3→0]
            order = [3, 2, 1, 0]
            values = [flipped[i] for i in order]
        else:
            values = flipped
        # Write new values back
        for (x, y), val in zip(block, values):
            new_grid[x][y] = val
    return new_grid


def draw_grid(screen, grid, N, cell_size):
    """Draw the grid on the pygame screen."""
    for i in range(N):
        for j in range(N):
            color = (255, 255, 255) if grid[i][j] == 1 else (0, 0, 0)
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            screen.fill(color, rect)


def main():
    parser = argparse.ArgumentParser(description="Block Cellular Automaton Simulation")
    parser.add_argument("--size", type=int, default=100,
                        help="Grid size N (must be even)")
    parser.add_argument("--prob", type=float, default=0.5,
                        help="Initial live-cell probability")
    parser.add_argument("--wrap", action="store_true",
                        help="Enable wraparound boundary conditions")
    args = parser.parse_args()

    N = args.size
    if N % 2 != 0:
        print("Error: Grid size must be an even number.")
        sys.exit(1)
    prob = args.prob
    wrap = args.wrap

    # Initialize pygame
    pygame.init()
    cell_size = 6
    screen = pygame.display.set_mode((N * cell_size, N * cell_size))
    pygame.display.set_caption("Block Cellular Automaton")
    clock = pygame.time.Clock()

    # Initialize grid
    grid = init_grid(N, prob)
    pattern = 'blue'

    running = True
    generation = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Draw current state
        draw_grid(screen, grid, N, cell_size)
        pygame.display.flip()

        # Compute next generation
        grid = apply_rules(grid, N, pattern, wrap)
        pattern = 'red' if pattern == 'blue' else 'blue'
        generation += 1

        # Control simulation speed (frames per second)
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
