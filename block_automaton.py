#!/usr/bin/env python3
"""
Block Automaton Simulation with Interactive Replay and Comparative Metrics

Features:
1. Interactive setup for grid parameters (size, probability, steps, wrap, delay).
2. Run automaton and collect metrics (density, stability, activity, block activity, perimeter).
3. Post-run menu:
   1) Plot last run metrics (individual graphs).
   2) Compare metrics across all runs (choose which metrics to plot).
   3) New run (re-enter setup).
4. Loop until user chooses to plot or compare, then exit.

Dependencies:
    pip install pygame numpy matplotlib
"""
import sys
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Display settings
default_cell_size = 5
MARGIN = 40
MAX_WINDOW_DIM = 600

# Input helpers
def ask_int(prompt, default):
    try: return int(input(prompt).strip() or default)
    except: return default

def ask_float(prompt, default):
    try: return float(input(prompt).strip() or default)
    except: return default

def ask_bool(prompt, default=False):
    val = input(prompt).strip().lower()
    if val in ('y','yes','1','true'): return True
    if val in ('n','no','0','false'): return False
    return default

# Automaton functions
def initialize_grid(N, prob):
    return np.random.choice([0,1], size=(N,N), p=[1-prob, prob])

def get_block_positions(N, gen, wrap):
    if gen % 2 == 1:
        starts = list(range(0, N, 2))
    else:
        if wrap:
            starts = [(1 + 2*k) % N for k in range((N+1)//2)]
        else:
            starts = list(range(1, N-1, 2))
    return [(i,j) for i in starts for j in starts]

def update_grid(grid, gen, wrap):
    N = grid.shape[0]
    new = grid.copy()
    for i,j in get_block_positions(N, gen, wrap):
        coords = [((i+di)%N, (j+dj)%N) for di in (0,1) for dj in (0,1)]
        block = np.array([grid[x,y] for x,y in coords]).reshape(2,2)
        s = block.sum()
        if s == 2: continue
        nb = 1 - block
        if s == 3: nb = np.rot90(nb, 2)
        for idx,(x,y) in enumerate(coords): new[x,y] = nb[idx//2, idx%2]
    return new

# Drawing functions
def draw_grid(screen, grid, cell_size, margin):
    N = grid.shape[0]
    for i in range(N):
        for j in range(N):
            color = (0,0,0) if grid[i,j] == 1 else (255,255,255)
            rect = pygame.Rect(j*cell_size, i*cell_size+margin, cell_size, cell_size)
            screen.fill(color, rect)

def draw_partitions(screen, gen, cell_size, margin, N, wrap):
    dash, gap = 5,5
    height, width = N*cell_size, N*cell_size
    if gen % 2 == 1:
        color, offs, dashed = (0,0,255), range(0,N,2), False
    else:
        color, offs, dashed = (255,0,0), range(1,N,2), True
    for off in offs:
        x = off*cell_size; y0,y1 = margin, margin+height
        if not dashed:
            pygame.draw.line(screen,color,(x,y0),(x,y1),2)
        else:
            y=y0
            while y<y1:
                y2 = min(y+dash, y1)
                pygame.draw.line(screen,color,(x,y),(x,y2),1)
                y += dash + gap
        y = margin + off*cell_size; x0,x1 = 0, width
        if not dashed:
            pygame.draw.line(screen,color,(x0,y),(x1,y),2)
        else:
            x=x0
            while x<x1:
                x2 = min(x+dash, x1)
                pygame.draw.line(screen,color,(x,y),(x2,y),1)
                x += dash + gap

# Metrics functions
def compute_block_activity(prev, grid, gen, wrap):
    blocks = get_block_positions(grid.shape[0], gen-1, wrap)
    if not blocks: return 0.0
    changed = sum(any(prev[x,y] != grid[x,y] for x,y in [((i+di)%grid.shape[0], (j+dj)%grid.shape[0])
                for di in (0,1) for dj in (0,1)]) for i,j in blocks)
    return changed/len(blocks)

def compute_perimeter(grid):
    N = grid.shape[0]; perim = 0
    for i in range(N):
        for j in range(N):
            perim += abs(int(grid[i,j]) - int(grid[i,(j+1)%N]))
            perim += abs(int(grid[i,j]) - int(grid[(i+1)%N,j]))
    return perim/(2*N*N)

# Single run collecting metrics
def simulate_run(params):
    N,prob,wrap,steps,cell_size,delay = params
    grid = initialize_grid(N,prob)
    prev = None
    metrics = {'density':[], 'stability':[], 'activity':[], 'block_act':[], 'perimeter':[]}
    pygame.init()
    if cell_size is None:
        usable = MAX_WINDOW_DIM - MARGIN
        cell_size = max(default_cell_size, usable//N)
    screen = pygame.display.set_mode((N*cell_size, N*cell_size+MARGIN))
    font = pygame.font.SysFont(None,24)
    clock = pygame.time.Clock(); gen = 1
    while gen <= steps:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return metrics
        d = grid.mean()
        s = (grid == prev).mean() if prev is not None else 0.0
        a = 1 - s
        ba = compute_block_activity(prev,grid,gen,wrap) if prev is not None else 0.0
        p = compute_perimeter(grid)
        metrics['density'].append(d)
        metrics['stability'].append(s)
        metrics['activity'].append(a)
        metrics['block_act'].append(ba)
        metrics['perimeter'].append(p)
        screen.fill((200,200,200),(0,0,N*cell_size,MARGIN))
        draw_grid(screen,grid,cell_size,MARGIN)
        draw_partitions(screen,gen,cell_size,MARGIN,N,wrap)
        header = f"Gen:{gen}/{steps}   Density:{d:.3f}   Stability:{s:.3f}"
        screen.blit(font.render(header,True,(0,0,0)),(5,5))
        pygame.display.flip()
        if delay > 0: pygame.time.delay(delay)
        clock.tick(60)
        prev = grid.copy(); grid = update_grid(grid,gen,wrap); gen += 1
    pygame.quit(); return metrics

# Plotting helpers
def plot_individual(metrics):
    for k,vals in metrics.items():
        plt.figure(k)
        plt.plot(range(1,len(vals)+1), vals)
        plt.title(k.replace('_',' ').title())
        plt.xlabel('Generation'); plt.ylabel(k)
    plt.show()

def plot_comparison(all_runs, selected_metrics):
    gens = range(1, len(all_runs[0][selected_metrics[0]])+1)
    for k in selected_metrics:
        plt.figure(k)
        for i,run in enumerate(all_runs):
            plt.plot(gens, run[k], label=f'Run {i+1}')
        plt.title(k.replace('_',' ').title())
        plt.xlabel('Generation'); plt.ylabel(k)
        plt.legend()
    plt.show()

# Main interactive loop
def main():
    all_runs = []
    while True:
        # Setup prompts
        print("\n=== Simulation Setup ===")
        N = ask_int("Grid size N×N (even) [100]: ", 100)
        if N % 2 != 0:
            N += 1; print(f"Adjusted to even N={N}")
        pct = ask_float("Alive probability % [50]: ", 50)
        prob = max(0.0, min(100.0, pct)) / 100.0
        steps = ask_int("Number of generations [250]: ", 250)
        wrap = ask_bool("Wrap-around? (y/N) [N]: ", False)
        delay = ask_int("Delay between gens (ms) [0]: ", 0)
        params = (N,prob,wrap,steps,None,delay)
        # Run simulation
        metrics = simulate_run(params)
        all_runs.append(metrics)
        # Post-run menu
        while True:
            print("\nRun complete. Options:\n1) Plot last run metrics\n2) Compare runs metrics\n3) New run")
            choice = input("Choose 1, 2 or 3: ").strip()
            if choice == '1':
                plot_individual(metrics)
                return
            elif choice == '2':
                if len(all_runs) < 2:
                    print("Need at least two runs to compare.")
                    continue
                print("Available metrics:", ', '.join(metrics.keys()))
                sel = input("Enter metric names comma-separated: ").split(',')
                selected = [m.strip() for m in sel if m.strip() in metrics]
                if selected:
                    plot_comparison(all_runs, selected)
                    return
                else:
                    print("No valid metrics selected.")
            elif choice == '3':
                break
            else:
                print("Invalid choice.")
        # loop back for new run

if __name__ == '__main__':
    main()
