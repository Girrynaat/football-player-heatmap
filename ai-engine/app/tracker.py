import cv2
import csv
import os
import math
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "test/football.webm"
MODEL_PATH = "yolo11s.pt"

OUTPUT_CSV = "results/player_positions.csv"
OUTPUT_VIDEO = "results/tracked_output.mp4"

CONFIDENCE = 0.05
IMAGE_SIZE = 1280

# Maximum distance allowed for matching
MAX_DISTANCE = 120

# How long to keep a missing player alive
MAX_MISSED_FRAMES = 25

# Number of detections needed before a track is confirmed
MIN_CONFIRMATIONS = 2


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width} x {height}")
print(f"FPS: {fps}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# VIDEO OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACK STORAGE
# ============================================================

player_tracks = {}

next_player_id = 1


# Each track contains:
#
# {
#     "position": (x, y),
#     "previous_position": (x, y),
#     "velocity": (vx, vy),
#     "missed": 0,
#     "confirmations": 2
# }


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time_seconds",
    "player_id",
    "center_x",
    "center_y",
    "confidence"
])


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

print("Starting improved player tracking...")


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    time_seconds = frame_number / fps


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=0,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # COLLECT DETECTIONS
    # ========================================================

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy()
            )

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            center_x = int((x1 + x2) / 2)

            # Bottom-center of bounding box
            center_y = int(y2)

            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "center": (center_x, center_y),
                "confidence": confidence
            })


    # ========================================================
    # INCREASE MISSED COUNT
    # ========================================================

    for player_id in player_tracks:

        player_tracks[player_id]["missed"] += 1


    # ========================================================
    # MATCH DETECTIONS TO TRACKS
    # ========================================================

    used_player_ids = set()

    for detection in detections:

        current_position = detection["center"]

        best_player_id = None
        best_distance = MAX_DISTANCE


        # ----------------------------------------------------
        # Search existing tracks
        # ----------------------------------------------------

        for player_id, track in player_tracks.items():

            if player_id in used_player_ids:
                continue


            # ------------------------------------------------
            # MOTION PREDICTION
            # ------------------------------------------------

            previous_position = track["position"]

            velocity = track["velocity"]

            predicted_position = (
                int(previous_position[0] + velocity[0]),
                int(previous_position[1] + velocity[1])
            )


            # ------------------------------------------------
            # Compare detection with predicted position
            # ------------------------------------------------

            d = distance(
                current_position,
                predicted_position
            )


            if d < best_distance:

                best_distance = d
                best_player_id = player_id


        # ====================================================
        # EXISTING PLAYER
        # ====================================================

        if best_player_id is not None:

            player_id = best_player_id

            track = player_tracks[player_id]


            # -----------------------------------------------
            # Calculate movement
            # -----------------------------------------------

            old_position = track["position"]

            velocity_x = (
                current_position[0] -
                old_position[0]
            )

            velocity_y = (
                current_position[1] -
                old_position[1]
            )


            # -----------------------------------------------
            # Update velocity
            # -----------------------------------------------

            track["velocity"] = (
                velocity_x,
                velocity_y
            )


            # -----------------------------------------------
            # Update position
            # -----------------------------------------------

            track["previous_position"] = old_position

            track["position"] = current_position

            track["missed"] = 0

            track["confirmations"] += 1

            used_player_ids.add(player_id)


        # ====================================================
        # NEW PLAYER
        # ====================================================

        else:

            player_id = next_player_id

            next_player_id += 1


            player_tracks[player_id] = {

                "position": current_position,

                "previous_position": current_position,

                "velocity": (0, 0),

                "missed": 0,

                "confirmations": 1
            }

            used_player_ids.add(player_id)


        # ====================================================
        # ONLY SAVE CONFIRMED TRACKS
        # ====================================================

        if (
            player_tracks[player_id]["confirmations"]
            >= MIN_CONFIRMATIONS
        ):

            csv_writer.writerow([
                frame_number,
                round(time_seconds, 3),
                player_id,
                current_position[0],
                current_position[1],
                round(
                    detection["confidence"],
                    3
                )
            ])


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (
                detection["x1"],
                detection["y1"]
            ),
            (
                detection["x2"],
                detection["y2"]
            ),
            (0, 255, 0),
            2
        )


        # ====================================================
        # DRAW PLAYER ID
        # ====================================================

        cv2.putText(
            frame,
            f"Player {player_id}",
            (
                detection["x1"],
                max(
                    20,
                    detection["y1"] - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DRAW POSITION
        # ====================================================

        cv2.circle(
            frame,
            current_position,
            5,
            (0, 0, 255),
            -1
        )


    # ========================================================
    # REMOVE VERY OLD TRACKS
    # ========================================================

    players_to_remove = []

    for player_id, track in player_tracks.items():

        if (
            track["missed"]
            > MAX_MISSED_FRAMES
        ):

            players_to_remove.append(
                player_id
            )


    for player_id in players_to_remove:

        del player_tracks[player_id]


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Active Tracks: {len(player_tracks)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    out.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
csv_file.close()


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("Improved tracking completed.")

print(
    f"CSV saved to: {OUTPUT_CSV}"
)

print(
    f"Video saved to: {OUTPUT_VIDEO}"
)

print(
    f"Total frames processed: {frame_number}"
)

print(
    f"Total IDs created: {next_player_id - 1}"
)