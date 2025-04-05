import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.2";
const videoElement = document.querySelector(".webcam");

let webcamScale = 0.5;

function updateWebcamScale(value) {
  requestAnimationFrame(() => {
    webcamScale = value;
    document.getElementById("webcamScaleValue").textContent = value;
    setupWebcam();
  });
}

function updateCanvasWidth(value) {
  requestAnimationFrame(() => {
    const canvas = document.getElementById("canvas");
    canvas.width = value;
    canvas.height = (value * 3) / 4; // Maintain aspect ratio
    document.getElementById("canvasWidthValue").textContent = value;
  });
}

document.getElementById("webcamScale").addEventListener("input", (e) => {
  updateWebcamScale(parseFloat(e.target.value));
});

document.getElementById("canvasWidth").addEventListener("input", (e) => {
  updateCanvasWidth(parseInt(e.target.value));
});

async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        frameRate: { ideal: 60 },
      },
    });
    videoElement.srcObject = stream;
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
}

const canvasElement = document.querySelector(".depth-estimation");
const canvasContext = canvasElement.getContext("2d");

// Assuming you have the Hugging Face Transformers pipeline for depth estimation loaded
async function estimateDepth() {
  console.log("Loading depth estimator...");
  const depthEstimator = await pipeline(
    "depth-estimation",
    "onnx-community/depth-anything-v2-small",
    { device: "webgpu", dtype: "fp32" }
  );

  let frameInProgress = false;
  let processedFrameCount = 0;
  let lastFpsUpdate = Date.now();

  // Add display FPS counter
  const displayFpsCounter = () => {
    const now = Date.now();
    const elapsed = now - lastFpsUpdate;
    if (elapsed >= 1000) {
      console.log({ processedFrameCount });
      const processingFps = ((processedFrameCount * 1000) / elapsed).toFixed(2);
      document.getElementById(
        "fpsCounter"
      ).textContent = `FPS: ${processingFps}`;
      processedFrameCount = 0;
      lastFpsUpdate = now;
    }
    requestAnimationFrame(displayFpsCounter);
  };

  displayFpsCounter();

  const offscreenCanvas = document.createElement("canvas");
  const offscreenContext = offscreenCanvas.getContext("2d");

  videoElement.addEventListener("play", () => {
    const processFrame = async () => {
      if (videoElement.paused || videoElement.ended) {
        console.log("Video paused");
        return;
      }
      if (frameInProgress) {
        console.log("Processing already in progress");
        return;
      }
      frameInProgress = true;
      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;

      // Prevents multiple dimensions being processed when user changes width
      const destionationWidth = canvasContext.canvas.width;
      const destionationHeight = canvasContext.canvas.height;

      const processingWidth = videoWidth * webcamScale;
      const processingHeight = videoHeight * webcamScale;

      // Create offscreen canvas for lower resolution processing
      offscreenCanvas.width = processingWidth;
      offscreenCanvas.height = processingHeight;

      // Draw the video frame at lower resolution
      offscreenContext.drawImage(
        videoElement,
        0,
        0,
        videoWidth,
        videoHeight,
        0,
        0,
        processingWidth,
        processingHeight
      );

      // Convert canvas to blob and create object URL
      const blob = await new Promise((resolve) =>
        offscreenCanvas.toBlob(resolve, "image/jpeg", 0.8)
      );
      const imageUrl = URL.createObjectURL(blob);
      const depthResponse = await depthEstimator(imageUrl);
      const depthData = depthResponse.depth.rgba().data;

      // Create ImageData and draw it to offscreen canvas
      const iData = new ImageData(
        new Uint8ClampedArray(depthData.buffer),
        processingWidth,
        processingHeight
      );
      offscreenContext.putImageData(iData, 0, 0);

      // Scale up and draw to main canvas
      canvasContext.drawImage(
        offscreenCanvas,
        0,
        0,
        processingWidth,
        processingHeight,
        0,
        0,
        destionationWidth,
        destionationHeight
      );

      // Update processed frame count after depth estimation
      processedFrameCount++;
      frameInProgress = false;
      requestAnimationFrame(processFrame);
    };

    processFrame();
  });
}

async function main() {
  await estimateDepth();
  console.log("Depth estimator loaded");
  setupWebcam();
}

main();

export { setupWebcam };
