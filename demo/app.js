const EMOTIONS = [
  "angry", "disgust", "fear", "happy",
  "neutral", "sad", "surprise",
];

const IMAGE_SIZE = 48;
const INFERENCE_INTERVAL_MS = 200;

let session = null;
let stream = null;
let intervalId = null;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusEl = document.getElementById("status");
const emotionEl = document.getElementById("emotion");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");

function buildBars() {
  barsEl.innerHTML = EMOTIONS.map((e) => `
    <div class="bar-row" data-emotion="${e}">
      <span class="bar-label">${e}</span>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
      <span class="bar-value">0%</span>
    </div>
  `).join("");
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function preprocessFrame() {
  const offscreen = document.createElement("canvas");
  offscreen.width = IMAGE_SIZE;
  offscreen.height = IMAGE_SIZE;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const imageData = offCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const float32 = new Float32Array(1 * 1 * IMAGE_SIZE * IMAGE_SIZE);

  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const gray = imageData.data[i * 4]; // R channel (grayscale)
    float32[i] = (gray / 255.0 - 0.5) / 0.5;
  }

  return new ort.Tensor("float32", float32, [1, 1, IMAGE_SIZE, IMAGE_SIZE]);
}

function updateUI(probs) {
  const maxIdx = probs.indexOf(Math.max(...probs));
  const maxProb = probs[maxIdx];

  emotionEl.textContent = EMOTIONS[maxIdx];
  confidenceEl.textContent = `confidence ${(maxProb * 100).toFixed(1)}%`;

  EMOTIONS.forEach((emotion, i) => {
    const row = barsEl.querySelector(`[data-emotion="${emotion}"]`);
    const fill = row.querySelector(".bar-fill");
    const value = row.querySelector(".bar-value");
    const pct = (probs[i] * 100).toFixed(0);
    fill.style.width = `${pct}%`;
    value.textContent = `${pct}%`;
  });
}

async function runInference() {
  if (!session || video.readyState < 2) return;

  try {
    const input = preprocessFrame();
    const results = await session.run({ image: input });
    const logits = Array.from(results.logits.data);
    const probs = softmax(logits);
    updateUI(probs);
  } catch (err) {
    console.error("Inference error:", err);
  }
}

async function loadModel() {
  statusEl.textContent = "Loading ONNX model…";
  session = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"],
  });
  statusEl.textContent = "Model ready. Click Start camera.";
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 320, height: 240 },
  });
  video.srcObject = stream;
  await video.play();

  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "Running inference…";

  intervalId = setInterval(runInference, INFERENCE_INTERVAL_MS);
}

function stopCamera() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Camera stopped.";
}

startBtn.addEventListener("click", () => startCamera().catch((e) => {
  statusEl.textContent = `Camera error: ${e.message}`;
}));
stopBtn.addEventListener("click", stopCamera);

buildBars();
loadModel().catch((e) => {
  statusEl.textContent = `Model load error: ${e.message}`;
});
