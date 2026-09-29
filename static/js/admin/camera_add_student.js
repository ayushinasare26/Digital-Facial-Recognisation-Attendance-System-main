// static/js/admin/camera_add_student.js - Admin Multi-Angle Face Enrollment & Embedding Extraction

const studentForm = document.getElementById("studentForm");
const saveInfoBtn = document.getElementById("saveInfoBtn");
const startCaptureBtn = document.getElementById("startCaptureBtn");
const addStudentBtn = document.getElementById("addStudentBtn");
const video = document.getElementById("video");
const captureStatus = document.getElementById("captureStatus");
const progressBar = document.getElementById("progressBar");
const enrollFeedback = document.getElementById("enrollFeedback");
const anglePromptBadge = document.getElementById("anglePromptBadge");
const enrollFaceGuide = document.getElementById("enrollFaceGuide");

const step2Badge = document.getElementById("step2Badge");
const step2Label = document.getElementById("step2Label");
const step3Badge = document.getElementById("step3Badge");
const step3Label = document.getElementById("step3Label");

let student_id = null;
let captured = 0;
const maxImages = 8;
let images = [];
let stream = null;

// Step 1: Save student basic info
if (studentForm) {
    studentForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        saveInfoBtn.disabled = true;
        saveInfoBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Registering...`;

        const fd = new FormData(studentForm);
        try {
            const res = await fetch("/admin/enroll", { method: "POST", body: fd });
            const j = await res.json();

            if (res.ok && j.student_id) {
                student_id = j.student_id;
                showFeedback("success", `Personal profile created for <strong>${j.name}</strong> (ID: #${j.student_id})! Proceed to camera capture.`);
                
                // Advance wizard indicator
                if (step2Badge) step2Badge.className = "badge bg-primary rounded-circle";
                if (step2Label) step2Label.className = "fw-bold small text-dark";

                startCaptureBtn.disabled = false;
                saveInfoBtn.className = "btn btn-outline-secondary rounded-pill px-4";
                saveInfoBtn.innerHTML = `<i class="bi bi-check-circle-fill text-success me-1"></i> Profile Registered`;
                if (anglePromptBadge) anglePromptBadge.innerText = "Click '2. Capture Reference Angles' to initialize camera.";
            } else {
                showFeedback("danger", j.error || "Failed to register student record.");
                saveInfoBtn.disabled = false;
                saveInfoBtn.innerHTML = `1. Save Info & Unlock Camera`;
            }
        } catch (err) {
            showFeedback("danger", "Server connection error: " + err.message);
            saveInfoBtn.disabled = false;
            saveInfoBtn.innerHTML = `1. Save Info & Unlock Camera`;
        }
    });
}

// Step 2: Start Camera and Capture Multi-Angle Face Frames
if (startCaptureBtn) {
    startCaptureBtn.addEventListener("click", async () => {
        startCaptureBtn.disabled = true;
        if (anglePromptBadge) anglePromptBadge.innerText = "Accessing optical camera...";

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            video.srcObject = stream;
            await video.play();
            if (enrollFaceGuide) enrollFaceGuide.classList.add("active");
            if (anglePromptBadge) anglePromptBadge.innerText = "Camera active. Starting multi-angle biometric sequence in 1s...";
            await new Promise(r => setTimeout(r, 1000));
            captureImagesLoop();
        } catch (err) {
            showFeedback("danger", "Webcam access error: " + err.message);
            startCaptureBtn.disabled = false;
            if (anglePromptBadge) anglePromptBadge.innerText = "Camera error. Please ensure permissions are granted and retry.";
        }
    });
}

const angleCues = [
    "Look straight at the camera (Front Angle)",
    "Look straight at the camera (Front Angle)",
    "Turn face slightly to the left",
    "Turn face slightly to the right",
    "Tilt face slightly upward",
    "Tilt face slightly downward",
    "Slight smile (Dynamic Expression)",
    "Final centered frame"
];

async function captureImagesLoop() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");

    images = [];
    captured = 0;

    while (captured < maxImages) {
        if (anglePromptBadge) anglePromptBadge.innerText = angleCues[captured] || "Hold still...";
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.92));
        images.push(blob);
        captured++;

        if (captureStatus) captureStatus.innerText = `${captured} / ${maxImages} frames captured`;
        if (progressBar) progressBar.style.width = `${(captured / maxImages) * 100}%`;

        await new Promise(r => setTimeout(r, 450));
    }

    // Step 3: Upload and Compute Embeddings on Server
    if (anglePromptBadge) anglePromptBadge.innerText = "Computing 128-d deep facial embeddings on server...";
    if (step3Badge) step3Badge.className = "badge bg-primary rounded-circle";
    if (step3Label) step3Label.className = "fw-bold small text-dark";

    const form = new FormData();
    form.append("student_id", student_id);
    images.forEach((b, i) => form.append("images[]", b, `angle_${i}.jpg`));

    try {
        const resp = await fetch("/admin/upload-face", { method: "POST", body: form });
        const j = await resp.json();

        if (resp.ok && j.success) {
            showFeedback("success", `🎉 ${j.message}`);
            if (anglePromptBadge) {
                anglePromptBadge.className = "badge bg-success-subtle text-success border border-success-subtle px-3 py-2 fs-6";
                anglePromptBadge.innerHTML = `<i class="bi bi-shield-check me-1"></i> Biometric Enrollment Complete (${j.embeddings_generated} deep vectors indexed)`;
            }
            if (addStudentBtn) addStudentBtn.disabled = false;
        } else {
            showFeedback("warning", j.error || "Enrollment encountered an issue with detected faces.");
            if (anglePromptBadge) anglePromptBadge.innerText = "Please ensure face is well lit and retry capture.";
            startCaptureBtn.disabled = false;
        }
    } catch (err) {
        showFeedback("danger", "Biometric processing failed: " + err.message);
        startCaptureBtn.disabled = false;
    } finally {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        if (enrollFaceGuide) enrollFaceGuide.classList.remove("active");
    }
}

if (addStudentBtn) {
    addStudentBtn.addEventListener("click", () => {
        if (student_id) {
            window.location.href = `/admin/students/${student_id}`;
        } else {
            window.location.href = "/admin/attendance";
        }
    });
}

function showFeedback(type, message) {
    if (enrollFeedback) {
        enrollFeedback.className = `alert alert-${type} rounded-3 py-2 px-3 small mb-3 shadow-sm`;
        enrollFeedback.innerHTML = message;
        enrollFeedback.classList.remove("d-none");
    }
}
