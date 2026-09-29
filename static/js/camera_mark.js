// camera_mark.js - Live Geo-Verified Facial Recognition & Liveness Attendance Script

const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const captureVerifyBtn = document.getElementById("captureVerifyBtn");
const markVideo = document.getElementById("markVideo");
const faceGuide = document.getElementById("faceGuide");
const challengeBanner = document.getElementById("challengeBanner");
const challengeProgress = document.getElementById("challengeProgress");
const scannerStatusText = document.getElementById("scannerStatusText");
const liveGpsBadge = document.getElementById("liveGpsBadge");
const demoModeToggle = document.getElementById("demoModeToggle");
const pipelineStateBadge = document.getElementById("pipelineStateBadge");

const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const resultPhoto = document.getElementById("resultPhoto");
const resultPersonName = document.getElementById("resultPersonName");
const resultConfidence = document.getElementById("resultConfidence");
const resultTimestamp = document.getElementById("resultTimestamp");
const resultAddress = document.getElementById("resultAddress");

const errorAlert = document.getElementById("errorAlert");
const errorAlertTitle = document.getElementById("errorAlertTitle");
const errorAlertMessage = document.getElementById("errorAlertMessage");

let videoStream = null;
let currentCoords = { latitude: null, longitude: null };
let isDemoMode = false;

// ---------------- Geolocation Initialization ----------------
function initGeolocation() {
    if ("geolocation" in navigator) {
        liveGpsBadge.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Locating...`;
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                currentCoords.latitude = pos.coords.latitude;
                currentCoords.longitude = pos.coords.longitude;
                liveGpsBadge.className = "badge bg-success-subtle text-success border border-success-subtle px-2 py-1 small";
                liveGpsBadge.innerHTML = `<i class="bi bi-geo-alt-fill"></i> ${currentCoords.latitude.toFixed(4)}°, ${currentCoords.longitude.toFixed(4)}°`;
            },
            (err) => {
                console.warn("Geolocation warning:", err.message);
                liveGpsBadge.className = "badge bg-warning-subtle text-warning border border-warning-subtle px-2 py-1 small";
                liveGpsBadge.innerHTML = `<i class="bi bi-exclamation-triangle"></i> GPS Denied/Offline`;
            },
            { enableHighAccuracy: true, timeout: 8000, maximumAge: 10000 }
        );
    } else {
        liveGpsBadge.className = "badge bg-danger-subtle text-danger border px-2 py-1 small";
        liveGpsBadge.innerHTML = `<i class="bi bi-x-circle"></i> No GPS API`;
    }
}

// ---------------- Demo Mode Toggle ----------------
demoModeToggle.addEventListener("change", (e) => {
    isDemoMode = e.target.checked;
    if (isDemoMode) {
        scannerStatusText.innerText = "Demo Mode Activated. Using mock camera selfie and mock GPS coordinates.";
        captureVerifyBtn.disabled = false;
        liveGpsBadge.className = "badge bg-info-subtle text-info border border-info-subtle px-2 py-1 small";
        liveGpsBadge.innerHTML = `<i class="bi bi-magic"></i> Mock GPS (19.0760°, 72.8777°)`;
        currentCoords = { latitude: 19.0760, longitude: 72.8777 };
    } else {
        scannerStatusText.innerText = "Camera standby. Click 'Start Camera' to initialize.";
        if (!videoStream) captureVerifyBtn.disabled = true;
        initGeolocation();
    }
});

// ---------------- Camera Controls ----------------
startCamBtn.addEventListener("click", async () => {
    startCamBtn.disabled = true;
    scannerStatusText.innerText = "Requesting webcam permissions...";
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
        });
        markVideo.srcObject = videoStream;
        await markVideo.play();

        faceGuide.classList.add("active");
        stopCamBtn.disabled = false;
        captureVerifyBtn.disabled = false;
        scannerStatusText.innerText = "Camera active. Please center your face inside the guide.";
        hideError();
    } catch (err) {
        console.error("Camera access error:", err);
        showError("Camera Access Error", "Unable to open webcam: " + err.message + ". Check browser permissions or enable Demo Mode.");
        startCamBtn.disabled = false;
    }
});

stopCamBtn.addEventListener("click", () => {
    stopCamera();
    scannerStatusText.innerText = "Camera stopped.";
    if (!isDemoMode) captureVerifyBtn.disabled = true;
    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
    faceGuide.classList.remove("active");
});

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
}

// ---------------- Capture Frame Helper ----------------
function grabFrameAsBlob() {
    const canvas = document.createElement("canvas");
    canvas.width = markVideo.videoWidth || 640;
    canvas.height = markVideo.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
    return new Promise(res => canvas.toBlob(res, "image/jpeg", 0.90));
}

// ---------------- Pipeline Stepper Animation Helper ----------------
const stageNames = [
    "Photo Intake",
    "Liveness Check",
    "Face Detection",
    "Embedding Extraction",
    "Embedding Gallery Match",
    "Geolocation Capture",
    "Reverse Geocoding",
    "Geotag Photo Stamping",
    "Audit Record Saved"
];

function resetStepper() {
    for (let i = 1; i <= 9; i++) {
        const el = document.getElementById(`stage-${i}`);
        if (!el) continue;
        el.className = "stage-card";
        el.querySelector(".stage-status").className = "badge bg-light text-muted border stage-status";
        el.querySelector(".stage-status").innerText = "Pending";
    }
    resultCard.style.display = "none";
    hideError();
}

function updateStage(index, status, customText = null) {
    const el = document.getElementById(`stage-${index}`);
    if (!el) return;
    const badge = el.querySelector(".stage-status");
    el.className = `stage-card ${status.toLowerCase()}`;
    
    if (status === "Completed") {
        badge.className = "badge bg-success text-white stage-status";
        badge.innerText = customText || "Completed";
    } else if (status === "Processing") {
        badge.className = "badge bg-primary text-white stage-status";
        badge.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Processing`;
    } else if (status === "Failed") {
        badge.className = "badge bg-danger text-white stage-status";
        badge.innerText = customText || "Failed";
    } else if (status === "Fallback") {
        badge.className = "badge bg-warning text-dark stage-status";
        badge.innerText = customText || "Fallback";
    } else if (status === "Skipped") {
        badge.className = "badge bg-secondary text-white stage-status";
        badge.innerText = "Skipped";
    }
}

// ---------------- Main Verification & Mark Attendance Flow ----------------
captureVerifyBtn.addEventListener("click", async () => {
    captureVerifyBtn.disabled = true;
    resetStepper();
    pipelineStateBadge.className = "badge bg-primary text-white";
    pipelineStateBadge.innerText = "Running Pipeline...";

    let primaryBlob = null;
    let livenessBlobs = [];

    // Stage 1 & 2: Liveness Challenge & Multi-Frame Intake
    if (videoStream) {
        challengeBanner.style.display = "block";
        challengeProgress.style.width = "20%";
        updateStage(1, "Processing");
        updateStage(2, "Processing");

        try {
            // First frame: Neutral face
            primaryBlob = await grabFrameAsBlob();
            challengeProgress.style.width = "40%";
            await new Promise(r => setTimeout(r, 250));

            // Second frame during blink / challenge
            challengeProgress.style.width = "75%";
            const f2 = await grabFrameAsBlob();
            livenessBlobs.push(f2);
            await new Promise(r => setTimeout(r, 250));

            // Third frame: Open face
            challengeProgress.style.width = "100%";
            const f3 = await grabFrameAsBlob();
            livenessBlobs.push(f3);
        } catch (e) {
            console.error("Frame capture error:", e);
        } finally {
            challengeBanner.style.display = "none";
            challengeProgress.style.width = "0%";
        }
    } else {
        updateStage(1, "Processing");
        updateStage(2, "Processing");
        await new Promise(r => setTimeout(r, 300));
    }

    // Build Form Payload
    const formData = new FormData();
    if (primaryBlob) {
        formData.append("image", primaryBlob, "selfie.jpg");
    }
    livenessBlobs.forEach((b, idx) => {
        formData.append("liveness_frames[]", b, `live_${idx}.jpg`);
    });
    
    if (currentCoords.latitude && currentCoords.longitude) {
        formData.append("latitude", currentCoords.latitude);
        formData.append("longitude", currentCoords.longitude);
    }
    formData.append("challenge_type", "blink");
    formData.append("is_demo", isDemoMode ? "true" : "false");

    // Call Backend 9-Stage Pipeline
    try {
        const resp = await fetch("/api/mark-attendance", {
            method: "POST",
            body: formData
        });
        const res = await resp.json();

        // Animate Stage Badges from backend stages response
        if (res.stages && Array.isArray(res.stages)) {
            res.stages.forEach((st, idx) => {
                const stageNum = idx + 1;
                updateStage(stageNum, st.status, st.status);
            });
        }

        if (res.success) {
            pipelineStateBadge.className = "badge bg-success text-white";
            pipelineStateBadge.innerText = "Attendance Verified";

            // Populate Result Card
            resultPersonName.innerText = `${res.student_name} (ID: #${res.student_id})`;
            resultConfidence.innerText = `${Math.round(res.confidence * 100)}% Match`;
            resultTimestamp.innerText = `📅 ${res.timestamp_display}`;
            resultAddress.innerText = res.address;
            
            if (res.photo_url) {
                resultPhoto.src = res.photo_url;
                resultPhoto.style.display = "block";
            } else {
                resultPhoto.style.display = "none";
            }

            if (res.status === "success") {
                resultBadge.className = "badge-verified";
                resultBadge.innerHTML = `<i class="bi bi-shield-check"></i> Verified`;
            } else {
                resultBadge.className = "badge-flagged";
                resultBadge.innerHTML = `<i class="bi bi-flag-fill"></i> Flagged for Review`;
            }

            resultCard.style.display = "block";
            resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } else {
            pipelineStateBadge.className = "badge bg-danger text-white";
            pipelineStateBadge.innerText = "Pipeline Interrupted";
            showError(res.error_stage || "Verification Failed", res.message || "Face or location criteria not met.");
        }
    } catch (err) {
        pipelineStateBadge.className = "badge bg-danger text-white";
        pipelineStateBadge.innerText = "Network Error";
        showError("Server Connection Error", "Unable to communicate with attendance engine: " + err.message);
    } finally {
        captureVerifyBtn.disabled = false;
    }
});

function showError(title, msg) {
    errorAlertTitle.innerText = title;
    errorAlertMessage.innerText = msg;
    errorAlert.style.display = "block";
}

function hideError() {
    errorAlert.style.display = "none";
}

// On page load, initialize geolocation
document.addEventListener("DOMContentLoaded", () => {
    initGeolocation();
});
