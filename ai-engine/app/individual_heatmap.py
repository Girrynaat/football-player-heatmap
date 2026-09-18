import csv
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==============================
# PATHS
# ==============================

CSV_PATH = "results/player_positions.csv"
VIDEO_PATH = "test/football.webm"

OUTPUT_DIR = "results/individual_heatmaps"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# READ VIDEO SIZE
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

print(f"Video resolution: {width} x {height}")


# ==============================
# READ CSV
# ==============================

players = {}

with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        player_id = int(row["player_id"])

        x = int(row["center_x"])
        y = int(row["center_y"])

        # Ignore positions outside video

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            continue

        if player_id not in players:

            players[player_id] = []

        players[player_id].append(
            (x, y)
        )


# ==============================
# SHOW PLAYER COUNTS
# ==============================

print(f"Players found: {len(players)}")

for player_id, positions in players.items():

    print(
        f"Player {player_id}: "
        f"{len(positions)} positions"
    )


# ==============================
# MINIMUM POSITIONS
# ==============================

MIN_POSITIONS = 5


# ==============================
# GENERATE HEATMAPS
# ==============================

generated = 0


for player_id, positions in players.items():

    # Skip players with very few detections

    if len(positions) < MIN_POSITIONS:

        continue


    print(
        f"\nGenerating heatmap for "
        f"Player {player_id}..."
    )


    # ------------------------------
    # CREATE EMPTY HEATMAP
    # ------------------------------

    heatmap = np.zeros(
        (height, width),
        dtype=np.float32
    )


    # ------------------------------
    # ADD PLAYER POSITIONS
    # ------------------------------

    for x, y in positions:

        heatmap[y, x] += 1


    # ------------------------------
    # SMOOTH
    # ------------------------------

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )


    # ------------------------------
    # NORMALIZE
    # ------------------------------

    if heatmap.max() > 0:

        heatmap = (
            heatmap /
            heatmap.max()
        )


    # ------------------------------
    # CREATE FIGURE
    # ------------------------------

    plt.figure(
        figsize=(12, 7)
    )


    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=[
            0,
            width,
            height,
            0
        ],
        interpolation="bilinear"
    )


    # ------------------------------
    # COLOR BAR
    # ------------------------------

    plt.colorbar(
        label="Relative Movement Density"
    )


    # ------------------------------
    # TITLE
    # ------------------------------

    plt.title(
        f"Player {player_id} Movement Heatmap"
    )


    plt.xlabel(
        "X Position"
    )

    plt.ylabel(
        "Y Position"
    )


    # ------------------------------
    # SAVE
    # ------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"player_{player_id}_heatmap.png"
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    generated += 1


# ==============================
# FINAL OUTPUT
# ==============================

print("\n====================================")
print("INDIVIDUAL HEATMAPS COMPLETED")
print("====================================")

print(
    f"Heatmaps generated: {generated}"
)

print(
    f"Output folder: {OUTPUT_DIR}"
)