#!/usr/bin/env python3
"""Generate a trajectory animation GIF for the robot navigation benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# Style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Load results
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

px = np.array(metrics['trajectory_x'])
py = np.array(metrics['trajectory_y'])
vx = np.array(metrics['velocity_x'])
vy = np.array(metrics['velocity_y'])
k = metrics['k']
N = metrics['N']

# Wall configuration (must match run.py)
WALL_X_MIN, WALL_X_MAX = 3.0, 7.0
WALL_Y_MIN, WALL_Y_MAX = 0.0, 3.0
R_ROBOT = 0.2
D_MARGIN = 0.1
D_SAFE = R_ROBOT + D_MARGIN
T_HORIZON = len(px)
DT = 0.4

# Colours
C_TRAJ = '#2166ac'
C_WALL = '#b2182b'
C_SAFE = '#2d6a4f'

fig, ax = plt.subplots(figsize=(8, 5.5))

def draw_static(ax):
    """Draw wall, safety zone, start/goal markers."""
    w = WALL_X_MAX - WALL_X_MIN
    h = WALL_Y_MAX - WALL_Y_MIN

    # Safety margin above wall
    safety_rect = Rectangle(
        (WALL_X_MIN, WALL_Y_MAX), w, D_SAFE,
        facecolor=C_WALL, alpha=0.12,
        edgecolor=C_WALL, linestyle='--', linewidth=0.8)
    ax.add_patch(safety_rect)

    # Wall body
    rect = Rectangle(
        (WALL_X_MIN, WALL_Y_MIN), w, h,
        facecolor=C_WALL, alpha=0.45,
        edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    cx = (WALL_X_MIN + WALL_X_MAX) / 2
    cy = (WALL_Y_MIN + WALL_Y_MAX) / 2
    ax.annotate('Wall', (cx, cy), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Start and goal
    ax.scatter([0.5], [2.5], c=C_SAFE, s=120, marker='o', zorder=7,
               edgecolors='black', linewidths=1.5, label='Start')
    ax.scatter([9.5], [2.5], c=C_WALL, s=150, marker='*', zorder=7,
               edgecolors='black', linewidths=0.8, label='Goal')

    ax.set_xlabel('$x$ position (m)')
    ax.set_ylabel('$y$ position (m)')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=9, loc='upper left')
    ax.set_xlim(-0.3, 10.5)
    ax.set_ylim(-0.3, 4.8)
    ax.set_aspect('equal')

draw_static(ax)

# Animated elements
trail_line, = ax.plot([], [], '-', color=C_TRAJ, linewidth=2, alpha=0.5, zorder=4)
robot_circle = Circle((px[0], py[0]), R_ROBOT, facecolor=C_TRAJ, alpha=0.7,
                       edgecolor='white', linewidth=1.5, zorder=8)
ax.add_patch(robot_circle)
vel_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
                        zorder=9)
title = ax.set_title(f'QP Robot Navigation ($t = 0.0$s)', fontsize=14)

# Info box
info_text = ax.text(0.98, 0.02,
                    f'$k = {k}$ active constraints\n$N = {N}$ scenarios',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

def init():
    trail_line.set_data([], [])
    robot_circle.center = (px[0], py[0])
    return trail_line, robot_circle, title

def update(frame):
    # Trail up to current position
    trail_line.set_data(px[:frame+1], py[:frame+1])

    # Robot position
    robot_circle.center = (px[frame], py[frame])

    # Velocity arrow
    scale = 0.3
    vel_arrow.xy = (px[frame] + vx[frame] * scale, py[frame] + vy[frame] * scale)
    vel_arrow.set_position((px[frame], py[frame]))

    t = frame * DT
    title.set_text(f'QP Robot Navigation ($t = {t:.1f}$s)')

    return trail_line, robot_circle, title

anim = FuncAnimation(fig, update, init_func=init,
                     frames=T_HORIZON, interval=200, blit=False)

out_path = os.path.join(benchmark_dir, 'results', 'trajectory_animation.gif')
anim.save(out_path, writer=PillowWriter(fps=5))
print(f'Saved: {out_path}')
plt.close()
