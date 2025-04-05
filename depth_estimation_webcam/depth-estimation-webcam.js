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
  setTimeout(() => {
    const canvas = document.getElementById("canvas");
    canvas.width = value;
    canvas.height = (value * 3) / 4; // Maintain aspect ratio
    document.getElementById("canvasWidthValue").textContent = value;
  }, 0);
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

      // Prevents multiple dimensions being processed when user changes width
      const destionationWidth = canvasContext.canvas.width;
      const destionationHeight = canvasContext.canvas.height;

      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;

      const processingWidth = videoWidth * webcamScale;
      const processingHeight = videoHeight * webcamScale;

      // console.log("Processing frame...");
      // Draw the current video frame to the canvas
      // Create an offscreen canvas to capture the frame
      const offscreenCanvas = document.createElement("canvas");
      offscreenCanvas.width = processingWidth;
      offscreenCanvas.height = processingHeight;
      const offscreenContext = offscreenCanvas.getContext("2d");

      // Draw the video frame directly at the lower resolution
      offscreenContext.drawImage(
        videoElement,
        0,
        0,
        videoWidth,
        videoHeight, // source dimensions
        0,
        0,
        processingWidth,
        processingHeight // destination dimensions
      );

      // Get the image data from the offscreen canvas
      const imageData = offscreenCanvas.toDataURL("image/jpeg", 0.8);

      // Run the Hugging Face pipeline to estimate depth
      const depthResponse = await depthEstimator(imageData);
      // Post-process and visualize the depth map
      const depthData = depthResponse.depth.rgba().data;

      // create ImageData instance
      const iData = new ImageData(
        new Uint8ClampedArray(depthData.buffer),
        processingWidth,
        processingHeight
      );
      canvasContext.putImageData(iData, 0, 0);

      // Scale up the result to destination size
      const resultCanvas = document.createElement("canvas");
      resultCanvas.width = processingWidth;
      resultCanvas.height = processingHeight;
      const resultContext = resultCanvas.getContext("2d");
      resultContext.putImageData(iData, 0, 0);

      canvasContext.drawImage(
        resultCanvas,
        0,
        0,
        processingWidth,
        processingHeight,
        0,
        0,
        destionationWidth,
        destionationHeight
      );
      // Continue processing frames
      requestAnimationFrame(processFrame);
      frameInProgress = false;
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
