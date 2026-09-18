const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

const app = express();

const PORT = 5000;

// AI Engine is running on a separate PC
const AI_URL = "http://10.72.189.216:8000/analyze";

// --------------------------------------------------
// Middleware
// --------------------------------------------------

app.use(cors());
app.use(express.json());

// --------------------------------------------------
// Multer configuration
// --------------------------------------------------

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/");
    },

    filename: (req, file, cb) => {
        const uniqueName = Date.now() + "-" + file.originalname;
        cb(null, uniqueName);
    }
});

const upload = multer({
    storage: storage
});

// --------------------------------------------------
// Test route
// --------------------------------------------------

app.get("/", (req, res) => {
    res.json({
        message: "Football Player Heatmap Backend is running"
    });
});

// --------------------------------------------------
// Health check
// --------------------------------------------------

app.get("/health", (req, res) => {
    res.json({
        status: "healthy"
    });
});

// --------------------------------------------------
// Video upload + AI analysis
// --------------------------------------------------

app.post("/api/analyze", upload.single("video"), async (req, res) => {

    try {

        // Check whether video was uploaded
        if (!req.file) {
            return res.status(400).json({
                status: "error",
                message: "No video file uploaded"
            });
        }

        console.log("----------------------------------------");
        console.log("Video received:", req.file.filename);
        console.log("Video path:", req.file.path);
        console.log("Sending video to AI engine...");
        console.log("----------------------------------------");

        // --------------------------------------------------
        // Create multipart form data for AI
        // AI expects the field name: "file"
        // --------------------------------------------------

        const formData = new FormData();

        formData.append(
            "file",
            fs.createReadStream(req.file.path)
        );

        // --------------------------------------------------
        // Send video to Python AI engine
        // --------------------------------------------------

        const aiResponse = await axios.post(
            AI_URL,
            formData,
            {
                headers: {
                    ...formData.getHeaders()
                },

                maxBodyLength: Infinity,
                maxContentLength: Infinity,

                // AI processing may take a long time
                timeout: 0
            }
        );

        console.log("----------------------------------------");
        console.log("AI response received successfully.");
        console.log("----------------------------------------");

        // --------------------------------------------------
        // Send AI result back to frontend
        // --------------------------------------------------

        res.json(aiResponse.data);

    } catch (error) {

        console.error("----------------------------------------");
        console.error("AI connection/processing error");
        console.error("----------------------------------------");

        // AI returned an error
        if (error.response) {

            console.error(
                "AI status:",
                error.response.status
            );

            console.error(
                "AI response:",
                error.response.data
            );

            return res.status(500).json({
                status: "error",
                message: "AI engine returned an error",
                aiError: error.response.data
            });
        }

        // Network/connection error
        console.error("Error:", error.message);

        return res.status(500).json({
            status: "error",
            message: "Could not connect to AI engine",
            error: error.message
        });
    }
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

// 0.0.0.0 allows other PCs on the same network
// to connect to this backend.
app.listen(PORT, "0.0.0.0", () => {

    console.log("----------------------------------------");
    console.log("Football Player Heatmap Backend");
    console.log("----------------------------------------");
    console.log(`Backend running on port ${PORT}`);
    console.log(`Backend URL: http://10.72.189.38:${PORT}`);
    console.log(`AI Engine URL: ${AI_URL}`);
    console.log("----------------------------------------");

});