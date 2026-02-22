console.log("JS LOADED");

const uploadBtn = document.getElementById("uploadBtn");
const videoInput = document.getElementById("videoInput");

uploadBtn.onclick = function () {
    videoInput.click();
};

videoInput.onchange = function () {
    const file = videoInput.files[0];
    if (!file) return;

    console.log("Video selected:", file.name);
    fakeProcessing();
};

const DEMO_TOTAL_TIME = 2000; // total fake time in ms (12 sec)
const DEMO_LOG_DELAY = 400;   // delay between each log line

function fakeProcessing() {

    uploadBtn.disabled = true;
    uploadBtn.innerText = "PROCESSING...";

    const overlay = document.getElementById("processingOverlay");
    const logsContainer = document.getElementById("processingLogs");
    const progressFill = document.getElementById("progressFill");

    logsContainer.innerHTML = "";
    progressFill.style.width = "0%";
    overlay.classList.remove("hidden");

    const logs = [
        "Initializing pipeline...",
        "Extracting frames...",
        "Running detection...",
        "Reconstructing depth map...",
        "Generating dense point cloud...",
        "Aligning to BIM reference...",
        "Optimizing geometry...",
        "Finalizing 3D mesh..."
    ];

    let currentProgress = 0;
    const progressStep = 100 / (DEMO_TOTAL_TIME / 100);

    // Smooth progress animation
    const progressInterval = setInterval(() => {
        currentProgress += progressStep;
        progressFill.style.width = Math.min(currentProgress, 100) + "%";

        if (currentProgress >= 100) {
            clearInterval(progressInterval);
        }
    }, 100);

    // Log timing
    logs.forEach((log, index) => {
        setTimeout(() => {
            const line = document.createElement("div");
            line.textContent = log;
            logsContainer.appendChild(line);
        }, index * DEMO_LOG_DELAY);
    });

    // Finish after total time
    setTimeout(() => {

        overlay.classList.add("hidden");
        uploadBtn.disabled = false;
        uploadBtn.innerText = "UPLOAD VIDEO";

        renderModel();

    }, DEMO_TOTAL_TIME);
}

function renderModel() {

    // 3D Model
    const viewer = document.getElementById("modelViewer");
    viewer.src = "/static/output/snow_pointcloud.glb";

    // Depth Map
    const depthImage = document.getElementById("depthImage");
    depthImage.src = "/static/output/depth_map.jpeg";
    depthImage.classList.remove("hidden");

    // Expected Image
    const expectedImage = document.getElementById("expectedImage");
    expectedImage.src = "/static/output/expected.png";
    expectedImage.classList.remove("hidden");

    // Actual Image
    const actualImage = document.getElementById("actualImage");
    actualImage.src = "/static/output/actual.png";
    actualImage.classList.remove("hidden");
}