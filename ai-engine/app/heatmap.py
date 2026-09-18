import os

# Use a non-GUI backend because this script runs from FastAPI
import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "results",
    "player_positions.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "team_heatmap.png"
)


# ============================================================
# VIDEO DIMENSIONS
# ============================================================

VIDEO_WIDTH = 898
VIDEO_HEIGHT = 506


# ============================================================
# DRAW FOOTBALL PITCH
# ============================================================

def draw_pitch(ax, width, height):

    # Pitch background
    ax.set_facecolor("#167a32")

    # Grass stripes
    stripe_width = width / 14

    for i in range(14):

        if i % 2 == 0:

            ax.add_patch(
                Rectangle(
                    (
                        i * stripe_width,
                        0
                    ),
                    stripe_width,
                    height,
                    facecolor="#1b8738",
                    alpha=0.35,
                    zorder=0
                )
            )

    # Outer boundary
    ax.add_patch(
        Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            edgecolor="white",
            linewidth=2.5,
            zorder=5
        )
    )

    center_x = width / 2
    center_y = height / 2

    # Halfway line
    ax.plot(
        [center_x, center_x],
        [0, height],
        color="white",
        linewidth=2,
        zorder=5
    )

    # Center circle
    center_radius = height * 0.18

    ax.add_patch(
        Circle(
            (center_x, center_y),
            center_radius,
            fill=False,
            edgecolor="white",
            linewidth=2,
            zorder=5
        )
    )

    # Center point
    ax.add_patch(
        Circle(
            (center_x, center_y),
            3,
            color="white",
            zorder=6
        )
    )

    # Penalty areas
    penalty_width = width * 0.12
    penalty_height = height * 0.48

    ax.add_patch(
        Rectangle(
            (
                0,
                center_y - penalty_height / 2
            ),
            penalty_width,
            penalty_height,
            fill=False,
            edgecolor="white",
            linewidth=2,
            zorder=5
        )
    )

    ax.add_patch(
        Rectangle(
            (
                width - penalty_width,
                center_y - penalty_height / 2
            ),
            penalty_width,
            penalty_height,
            fill=False,
            edgecolor="white",
            linewidth=2,
            zorder=5
        )
    )

    # Goal areas
    goal_width = width * 0.055
    goal_height = height * 0.22

    ax.add_patch(
        Rectangle(
            (
                0,
                center_y - goal_height / 2
            ),
            goal_width,
            goal_height,
            fill=False,
            edgecolor="white",
            linewidth=2,
            zorder=5
        )
    )

    ax.add_patch(
        Rectangle(
            (
                width - goal_width,
                center_y - goal_height / 2
            ),
            goal_width,
            goal_height,
            fill=False,
            edgecolor="white",
            linewidth=2,
            zorder=5
        )
    )

    # Penalty spots
    penalty_spot_distance = width * 0.095

    ax.add_patch(
        Circle(
            (
                penalty_spot_distance,
                center_y
            ),
            3,
            color="white",
            zorder=6
        )
    )

    ax.add_patch(
        Circle(
            (
                width - penalty_spot_distance,
                center_y
            ),
            3,
            color="white",
            zorder=6
        )
    )

    # Corner arcs
    corner_radius = 15

    ax.add_patch(
        Arc(
            (0, 0),
            corner_radius * 2,
            corner_radius * 2,
            theta1=0,
            theta2=90,
            color="white",
            linewidth=2,
            zorder=5
        )
    )

    ax.add_patch(
        Arc(
            (width, 0),
            corner_radius * 2,
            corner_radius * 2,
            theta1=90,
            theta2=180,
            color="white",
            linewidth=2,
            zorder=5
        )
    )

    ax.add_patch(
        Arc(
            (0, height),
            corner_radius * 2,
            corner_radius * 2,
            theta1=270,
            theta2=360,
            color="white",
            linewidth=2,
            zorder=5
        )
    )

    ax.add_patch(
        Arc(
            (width, height),
            corner_radius * 2,
            corner_radius * 2,
            theta1=180,
            theta2=270,
            color="white",
            linewidth=2,
            zorder=5
        )
    )


# ============================================================
# LOAD CSV
# ============================================================

print("Loading player positions...")

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(
    f"Position records: {len(df)}"
)

print(
    f"Unique tracking IDs: {df['player_id'].nunique()}"
)


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap = np.zeros(
    (
        VIDEO_HEIGHT,
        VIDEO_WIDTH
    ),
    dtype=np.float32
)


# ============================================================
# ADD POSITIONS
# ============================================================

for _, row in df.iterrows():

    x = int(row["center_x"])
    y = int(row["center_y"])

    if (
        0 <= x < VIDEO_WIDTH
        and
        0 <= y < VIDEO_HEIGHT
    ):

        heatmap[y, x] += 1


# ============================================================
# SMOOTH
# ============================================================

heatmap = cv2.GaussianBlur(
    heatmap,
    (0, 0),
    sigmaX=25,
    sigmaY=25
)


# ============================================================
# NORMALIZE
# ============================================================

if heatmap.max() > 0:

    heatmap = (
        heatmap /
        heatmap.max()
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)


# ============================================================
# DRAW PITCH
# ============================================================

draw_pitch(
    ax,
    VIDEO_WIDTH,
    VIDEO_HEIGHT
)


# ============================================================
# HEATMAP OVERLAY
# ============================================================

ax.imshow(
    heatmap,
    extent=[
        0,
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        0
    ],
    cmap="jet",
    alpha=0.60,
    interpolation="bilinear",
    zorder=2
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Player Heatmap - Team",
    fontsize=22,
    fontweight="bold",
    color="white",
    pad=15
)


# ============================================================
# COLORBAR
# ============================================================

sm = plt.cm.ScalarMappable(
    cmap="jet"
)

sm.set_array(heatmap)

cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.025,
    pad=0.02
)

cbar.set_label(
    "Activity",
    color="white",
    fontsize=12
)

cbar.ax.tick_params(
    colors="white"
)


# ============================================================
# AXIS
# ============================================================

ax.set_xlim(
    0,
    VIDEO_WIDTH
)

ax.set_ylim(
    VIDEO_HEIGHT,
    0
)

ax.set_aspect("equal")

ax.axis("off")


# ============================================================
# SAVE
# ============================================================

print(
    f"Saving heatmap to: {OUTPUT_PATH}"
)

fig.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
    facecolor="#167a32"
)

plt.close(fig)


# ============================================================
# VERIFY FILE
# ============================================================

if not os.path.exists(OUTPUT_PATH):

    raise RuntimeError(
        "Heatmap file was not created."
    )


print()
print(
    "Football pitch heatmap generated successfully."
)

print(
    f"Saved to: {OUTPUT_PATH}"
)