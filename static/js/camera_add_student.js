// camera_add_student.js - Multi-Angle Face Enrollment & Embedding Extraction

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
const maxImages = 10;
let images = [];
let stream = null;

// Step 1: Save student basic info
studentForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    saveInfoBtn.disabled = true;
    saveInfoBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Saving...`;

    const fd = new FormData(studentForm);
    try {
        const res = await fetch("/add_student", { method: "POST", body: fd });
        const j = await res.json();

        if (res.ok && j.student_id) {
            student_id = j.student_id;
            showFeedback("success", `Personal details saved for ${j.name}! Proceed to step 2.`);
            
            // Advance wizard indicator
            step2Badge.className = "badge bg-primary rounded-circle";
            step2Label.className = "fw-bold small text-dark";

            startCaptureBtn.disabled = false;
            saveInfoBtn.className = "btn btn-outline-secondary rounded-pill px-4";
            saveInfoBtn.innerHTML = `<i class="bi bi-check-circle-fill text-success me-1"></i> Info Saved`;
            anglePromptBadge.innerText = "Click '2. Capture Reference Angles' to turn on camera.";
        } else {
            showFeedback("danger", j.error || "Failed to save student details.");
            saveInfoBtn.disabled = false;
            saveInfoBtn.innerHTML = `1. Save Info & Unlock Camera`;
        }
    } catch (err) {
        showFeedback("danger", "Connection error: " + err.message);
        saveInfoBtn.disabled = false;
        saveInfoBtn.innerHTML = `1. Save Info & Unlock Camera`;
    }
});

// Step 2: Start Camera and Capture Multi-Angle Face Frames
startCaptureBtn.addEventListener("click", async () => {
    startCaptureBtn.disabled = true;
    anglePromptBadge.innerText = "Opening webcam stream...";

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
        });
        video.srcObject = stream;
        await video.play();
        enrollFaceGuide.classList.add("active");
        anglePromptBadge.innerText = "Camera active. Starting multi-angle capture sequence in 1s...";
        await new Promise(r => setTimeout(r, 1000));
        captureImagesLoop();
    } catch (err) {
        showFeedback("danger", "Webcam access error: " + err.message);
        startCaptureBtn.disabled = false;
        anglePromptBadge.innerText = "Camera error. Please allow permissions and retry.";
    }
});

const angleCues = [
    "Look straight at the camera (Front Angle)",
    "Look straight at the camera (Front Angle)",
    "Tilt head slightly to the left",
    "Tilt head slightly to the left",
    "Tilt head slightly to the right",
    "Tilt head slightly to the right",
    "Slight upward angle",
    "Slight downward angle",
    "Smile naturally",
    "Final centering check"
];

async function captureImagesLoop() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");

    images = [];
    captured = 0;

    while (captured < maxImages) {
        anglePromptBadge.innerText = angleCues[captured] || "Hold still...";
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.92));
        images.push(blob);
        captured++;

        captureStatus.innerText = `${captured} / ${maxImages} photos`;
        progressBar.style.width = `${(captured / maxImages) * 100}%`;

        // Wait between angle captures
        await new Promise(r => setTimeout(r, 400));
    }

    // Step 3: Upload and Compute Embeddings Immediately
    anglePromptBadge.innerText = "Extracting 128-d deep ResNet face embeddings on server...";
    step3Badge.className = "badge bg-primary rounded-circle";
    step3Label.className = "fw-bold small text-dark";

    const form = new FormData();
    form.append("student_id", student_id);
    images.forEach((b, i) => form.append("images[]", b, `angle_${i}.jpg`));

    try {
        const resp = await fetch("/upload_face", { method: "POST", body: form });
        const j = await resp.json();

        if (resp.ok && j.success) {
            showFeedback("success", `🎉 ${j.message}`);
            anglePromptBadge.className = "badge bg-success-subtle text-success border border-success-subtle px-3 py-2 fs-6";
            anglePromptBadge.innerHTML = `<i class="bi bi-shield-check me-1"></i> Biometric Enrollment Complete (${j.embeddings_generated} deep vectors stored)`;
            addStudentBtn.disabled = false;
        } else {
            showFeedback("warning", j.error || "Enrollment encountered an issue with detected faces.");
            anglePromptBadge.innerText = "Please retry capture with better lighting.";
            startCaptureBtn.disabled = false;
        }
    } catch (err) {
        showFeedback("danger", "Upload failed: " + err.message);
        startCaptureBtn.disabled = false;
    } finally {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        enrollFaceGuide.classList.remove("active");
    }
}

addStudentBtn.addEventListener("click", () => {
    window.location.href = "/students_page";
});

function showFeedback(type, message) {
    enrollFeedback.className = `alert alert-${type} rounded-3 py-2 px-3 small mb-3`;
    enrollFeedback.innerHTML = message;
    enrollFeedback.classList.remove("d-none");
}
