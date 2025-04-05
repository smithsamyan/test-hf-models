import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.2";
const videoElement = document.querySelector(".webcam");

async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
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

      // console.log("Processing frame...");
      // Draw the current video frame to the canvas
      // Create an offscreen canvas to capture the frame
      const offscreenCanvas = document.createElement("canvas");
      offscreenCanvas.width = videoElement.videoWidth;
      offscreenCanvas.height = videoElement.videoHeight;
      const offscreenContext = offscreenCanvas.getContext("2d");

      // Draw the current video frame to the offscreen canvas
      offscreenContext.drawImage(videoElement, 0, 0);

      // Get the image data from the offscreen canvas
      const imageData = offscreenCanvas.toDataURL("image/jpeg");

      // Run the Hugging Face pipeline to estimate depth
      const depthResponse = await depthEstimator(imageData);
      // Post-process and visualize the depth map
      const depthData = depthResponse.depth.rgba().data;

      // create ImageData instance
      const iData = new ImageData(
        new Uint8ClampedArray(depthData.buffer),
        640,
        480,
        { colorSpace: "srgb" }
      );
      canvasContext.putImageData(iData, 0, 0);

      // Continue processing frames
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
