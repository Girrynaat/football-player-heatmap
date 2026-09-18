import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "trajectories"
)

MIN_POSITIONS = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

print("Loading player positions...")

df = pd.read_csv(CSV_PATH)

print(f"Total position records: {len(df)}")
print(f"Total tracking IDs: {df['player_id'].nunique()}")


# ============================================================
# VIDEO DIMENSIONS
# ============================================================

VIDEO_WIDTH = 898
VIDEO_HEIGHT = 506


# ============================================================
# FIND PLAYERS
# ============================================================

player_ids = sorted(
    df["player_id"].unique()
)


print()
print("Generating player trajectories...")


generated = 0
skipped = 0


# ============================================================
# INDIVIDUAL PLAYER TRAJECTORIES
# ============================================================

for player_id in player_ids:

    player_data = df[
        df["player_id"] == player_id
    ].sort_values("frame")


    # --------------------------------------------------------
    # Ignore very short tracks
    # --------------------------------------------------------

    if len(player_data) < MIN_POSITIONS:

        skipped += 1

        continue


    # --------------------------------------------------------
    # Get coordinates
    # --------------------------------------------------------

    x = player_data["center_x"].values
    y = player_data["center_y"].values


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )


    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    ax.plot(
        x,
        y,
        linewidth=2,
        marker="o",
        markersize=3
    )


    # --------------------------------------------------------
    # Mark starting position
    # --------------------------------------------------------

    ax.scatter(
        x[0],
        y[0],
        s=100,
        marker="o",
        label="Start"
    )


    # --------------------------------------------------------
    # Mark ending position
    # --------------------------------------------------------

    ax.scatter(
        x[-1],
        y[-1],
        s=100,
        marker="X",
        label="End"
    )


    # --------------------------------------------------------
    # Add title
    # --------------------------------------------------------

    ax.set_title(
        f"Player {player_id} Movement Trajectory",
        fontsize=16,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "X Position (pixels)"
    )

    ax.set_ylabel(
        "Y Position (pixels)"
    )


    # --------------------------------------------------------
    # Keep video coordinate orientation
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        VIDEO_WIDTH
    )

    ax.set_ylim(
        VIDEO_HEIGHT,
        0
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"player_{player_id}_trajectory.png"
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    generated += 1


    print(
    f"Player {player_id}: "
    f"{len(player_data)} positions -> "
    f"{output_path}"
)


# ============================================================
# COMBINED TRAJECTORY
# ============================================================

print()
print("Generating combined trajectory...")


fig, ax = plt.subplots(
    figsize=(14, 8)
)


for player_id in player_ids:

    player_data = df[
        df["player_id"] == player_id
    ].sort_values("frame")


    if len(player_data) < MIN_POSITIONS:

        continue


    x = player_data["center_x"].values
    y = player_data["center_y"].values


    ax.plot(
        x,
        y,
        linewidth=1.5,
        alpha=0.7,
        label=f"Player {player_id}"
    )


# ============================================================
# COMBINED TITLE
# ============================================================

ax.set_title(
    "Player Movement Trajectories",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "X Position (pixels)"
)

ax.set_ylabel(
    "Y Position (pixels)"
)


# ============================================================
# VIDEO COORDINATES
# ============================================================

ax.set_xlim(
    0,
    VIDEO_WIDTH
)

ax.set_ylim(
    VIDEO_HEIGHT,
    0
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    alpha=0.3
)


# ============================================================
# SAVE COMBINED TRAJECTORY
# ============================================================

combined_path = os.path.join(
    OUTPUT_DIR,
    "all_player_trajectories.png"
)


plt.savefig(
    combined_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("Trajectory generation completed.")

print(
    f"Individual trajectories generated: {generated}"
)

print(
    f"Short tracks skipped: {skipped}"
)

print(
    f"Combined trajectory: {combined_path}"
)