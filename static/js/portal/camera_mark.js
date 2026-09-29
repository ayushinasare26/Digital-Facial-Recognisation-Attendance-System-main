// static/js/portal/camera_mark.js - Live Geo-Verified Facial Recognition & Liveness for User Portal

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
if (demoModeToggle) {
    demoModeToggle.addEventListener("change", (e) => {
        isDemoMode = e.target.checked;
        if (isDemoMode) {
            scannerStatusText.innerText = "Demo GPS Activated (Mock Coordinates). Click 'Start Camera' to capture your live face.";
            liveGpsBadge.className = "badge bg-info-subtle text-info border border-info-subtle px-2 py-1 small";
            liveGpsBadge.innerHTML = `<i class="bi bi-magic"></i> Mock GPS (19.0760°, 72.8777°)`;
            currentCoords = { latitude: 19.0760, longitude: 72.8777 };
            if (videoStream) captureVerifyBtn.disabled = false;
        } else {
            scannerStatusText.innerText = "Camera standby. Click 'Start Camera' to initialize.";
            if (!videoStream) captureVerifyBtn.disabled = true;
            initGeolocation();
        }
    });
}

// ---------------- Camera Controls ----------------
if (startCamBtn) {
    startCamBtn.addEventListener("click", async () => {
        startCamBtn.disabled = true;
        scannerStatusText.innerText = "Requesting webcam permissions...";
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            markVideo.srcObject = videoStream;
            await markVideo.play();

            if (faceGuide) faceGuide.classList.add("active");
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
}

if (stopCamBtn) {
    stopCamBtn.addEventListener("click", () => {
        stopCamera();
        scannerStatusText.innerText = "Camera stopped.";
        if (!isDemoMode) captureVerifyBtn.disabled = true;
        startCamBtn.disabled = false;
        stopCamBtn.disabled = true;
        if (faceGuide) faceGuide.classList.remove("active");
    });
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
        videoStream = null;
    }
    if (markVideo) markVideo.srcObject = null;
}

// ---------------- Frame Capture Utility ----------------
async function grabFrameAsBlob() {
    const canvas = document.createElement("canvas");
    canvas.width = markVideo.videoWidth || 640;
    canvas.height = markVideo.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
    });
}

// ---------------- Pipeline Stage Visualizer ----------------
function resetStages() {
    for (let i = 1; i <= 9; i++) {
        const badge = document.getElementById(`stage-badge-${i}`);
        if (badge) {
            badge.className = "stage-pill stage-waiting";
            badge.innerHTML = `<span class="badge-dot"></span> Waiting`;
        }
    }
}

function updateStage(num, statusText, stateClass) {
    const badge = document.getElementById(`stage-badge-${num}`);
    if (!badge) return;

    badge.className = "stage-pill";
    if (stateClass === "Passed" || stateClass === "completed" || stateClass === "success") {
        badge.classList.add("stage-passed");
        badge.innerHTML = `<i class="bi bi-check-circle-fill me-1"></i> ${statusText || "Passed"}`;
    } else if (stateClass === "Failed" || stateClass === "failed") {
        badge.classList.add("stage-failed");
        badge.innerHTML = `<i class="bi bi-x-circle-fill me-1"></i> ${statusText || "Failed"}`;
    } else if (stateClass === "Flagged" || stateClass === "flagged") {
        badge.classList.add("stage-flagged");
        badge.innerHTML = `<i class="bi bi-flag-fill me-1"></i> ${statusText || "Flagged"}`;
    } else {
        badge.classList.add("stage-active");
        badge.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Processing`;
    }
}

// ---------------- Attendance Verification Trigger ----------------
if (captureVerifyBtn) {
    captureVerifyBtn.addEventListener("click", async () => {
        hideError();
        if (resultCard) resultCard.style.display = "none";
        resetStages();

        captureVerifyBtn.disabled = true;
        startCamBtn.disabled = true;
        if (stopCamBtn) stopCamBtn.disabled = true;

        if (pipelineStateBadge) {
            pipelineStateBadge.className = "badge bg-warning text-dark";
            pipelineStateBadge.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Verifying...`;
        }

        let primaryBlob = null;
        let livenessBlobs = [];

        // 1. Capture primary frame & burst frames for liveness from live camera
        if (videoStream) {
            try {
                if (challengeBanner) challengeBanner.style.display = "block";
                if (challengeProgress) challengeProgress.style.width = "25%";

                updateStage(1, "Capturing Face", "active");
                primaryBlob = await grabFrameAsBlob();
                livenessBlobs.push(primaryBlob);
                await new Promise((r) => setTimeout(r, 250));

                if (challengeProgress) challengeProgress.style.width = "70%";
                const f2 = await grabFrameAsBlob();
                livenessBlobs.push(f2);
                await new Promise((r) => setTimeout(r, 250));

                if (challengeProgress) challengeProgress.style.width = "100%";
                const f3 = await grabFrameAsBlob();
                livenessBlobs.push(f3);
            } catch (e) {
                console.error("Frame capture error:", e);
            } finally {
                if (challengeBanner) challengeBanner.style.display = "none";
                if (challengeProgress) challengeProgress.style.width = "0%";
            }
        } else if (!isDemoMode) {
            showError("Camera Standby", "Please click 'Start Camera' first to capture your real live face.");
            captureVerifyBtn.disabled = false;
            startCamBtn.disabled = false;
            if (pipelineStateBadge) {
                pipelineStateBadge.className = "badge bg-secondary text-white";
                pipelineStateBadge.innerText = "Camera Required";
            }
            return;
        } else {
            updateStage(1, "Processing", "active");
            updateStage(2, "Processing", "active");
            await new Promise((r) => setTimeout(r, 300));
        }

        // Build Payload
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

        // Call User Portal Endpoint
        try {
            const resp = await fetch("/portal/mark-attendance", {
                method: "POST",
                body: formData
            });
            const res = await resp.json();

            // Animate Stages from pipeline result
            if (res.stages && Array.isArray(res.stages)) {
                res.stages.forEach((st, idx) => {
                    const stageNum = idx + 1;
                    updateStage(stageNum, st.status, st.status);
                });
            }

            if (resp.ok && res.success) {
                if (pipelineStateBadge) {
                    pipelineStateBadge.className = "badge bg-success text-white";
                    pipelineStateBadge.innerText = "Attendance Verified";
                }

                if (resultPersonName) resultPersonName.innerText = `${res.student_name} (ID: #${res.student_id})`;
                if (resultConfidence) resultConfidence.innerText = `${Math.round(res.confidence * 100)}% Match`;
                if (resultTimestamp) resultTimestamp.innerText = `📅 ${res.timestamp_display || new Date().toLocaleString()}`;
                if (resultAddress) resultAddress.innerText = res.address || "Location Verified";

                if (resultPhoto && res.photo_url) {
                    resultPhoto.src = res.photo_url;
                    resultPhoto.style.display = "block";
                }

                if (resultBadge) {
                    if (res.status === "success") {
                        resultBadge.className = "badge bg-success text-white px-3 py-2 rounded-pill";
                        resultBadge.innerHTML = `<i class="bi bi-shield-check"></i> Verified`;
                    } else {
                        resultBadge.className = "badge bg-warning text-dark px-3 py-2 rounded-pill";
                        resultBadge.innerHTML = `<i class="bi bi-flag-fill"></i> Flagged for Review`;
                    }
                }

                if (resultCard) {
                    resultCard.style.display = "block";
                    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
                }
            } else {
                if (pipelineStateBadge) {
                    pipelineStateBadge.className = "badge bg-danger text-white";
                    pipelineStateBadge.innerText = "Verification Failed";
                }
                showError(
                    res.error_stage || "Verification Failed",
                    res.message || "Attendance could not be recorded. Please ensure your face is clearly framed and try again."
                );
            }
        } catch (err) {
            console.error("Attendance submission error:", err);
            if (pipelineStateBadge) {
                pipelineStateBadge.className = "badge bg-danger text-white";
                pipelineStateBadge.innerText = "Network Error";
            }
            showError("Network Error", "Unable to communicate with verification server: " + err.message);
        } finally {
            captureVerifyBtn.disabled = false;
            startCamBtn.disabled = false;
            if (stopCamBtn) stopCamBtn.disabled = !videoStream;
        }
    });
}

// ---------------- Error Helpers ----------------
function showError(title, message) {
    if (errorAlert) {
        if (errorAlertTitle) errorAlertTitle.innerText = title;
        if (errorAlertMessage) errorAlertMessage.innerText = message;
        errorAlert.style.display = "block";
        errorAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function hideError() {
    if (errorAlert) errorAlert.style.display = "none";
}

// Auto init geolocation
initGeolocation();
