import os
import shutil
import subprocess
import sys

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Football Player Heatmap API",
    description="AI API for football player movement analysis",
    version="1.0.0"
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
TEST_DIR = "test"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


# ============================================================
# SERVE RESULT FILES
# ============================================================

app.mount(
    "/results",
    StaticFiles(directory=RESULT_DIR),
    name="results"
)


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Football Player Heatmap API is running",
        "status": "ok"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# ANALYZE FOOTBALL VIDEO
# ============================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    try:

        # ====================================================
        # 1. SAVE UPLOADED VIDEO
        # ====================================================

        filename = file.filename

        input_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        with open(
            input_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ====================================================
        # 2. COPY VIDEO TO TEST FOLDER
        # ====================================================

        test_video = os.path.join(
            TEST_DIR,
            "football.webm"
        )

        shutil.copy(
            input_path,
            test_video
        )


        # ====================================================
        # 3. RUN PLAYER TRACKING
        # ====================================================

        print("Starting player tracking...")

        tracking = subprocess.run(
            [
                sys.executable,
                "app/tracker.py"
            ],
            capture_output=True,
            text=True
        )


        if tracking.returncode != 0:

            return {
                "status": "error",
                "stage": "tracking",
                "error": tracking.stderr
            }


        # ====================================================
        # 4. GENERATE TEAM HEATMAP
        # ====================================================

        print("Generating team heatmap...")

        heatmap = subprocess.run(
            [
                sys.executable,
                "app/heatmap.py"
            ],
            capture_output=True,
            text=True
        )


        if heatmap.returncode != 0:

            return {
                "status": "error",
                "stage": "heatmap",
                "error": heatmap.stderr
            }


        # ====================================================
        # 5. GENERATE TRAJECTORIES
        # ====================================================

        print("Generating player trajectories...")

        trajectory = subprocess.run(
            [
                sys.executable,
                "app/trajectory.py"
            ],
            capture_output=True,
            text=True
        )


        if trajectory.returncode != 0:

            return {
                "status": "error",
                "stage": "trajectory",
                "error": trajectory.stderr
            }


        # ====================================================
        # 6. FIND TRAJECTORY FILES
        # ====================================================

        trajectory_dir = os.path.join(
            RESULT_DIR,
            "trajectories"
        )

        trajectory_files = []


        if os.path.exists(
            trajectory_dir
        ):

            for filename in os.listdir(
                trajectory_dir
            ):

                if filename.endswith(".png"):

                    trajectory_files.append(
                        filename
                    )


        # ====================================================
        # 7. RETURN RESULTS
        # ====================================================

        return {

            "status": "completed",

            "message":
                "Football video analyzed successfully.",

            "files": {

                "player_positions":
                    "/results/player_positions.csv",

                "team_heatmap":
                    "/results/team_heatmap.png",

                "tracked_video":
                    "/results/tracked_output.mp4",

                "trajectories":
                    [
                        f"/results/trajectories/{filename}"
                        for filename
                        in trajectory_files
                    ]
            }

        }


    except Exception as e:

        return {

            "status": "error",

            "stage": "api",

            "error": str(e)

        }