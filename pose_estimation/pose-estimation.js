import {
  AutoModel,
  AutoImageProcessor,
  RawImage,
} from "@huggingface/transformers";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const currentDir = path.dirname(fileURLToPath(import.meta.url));

// Load model and processor
const model_id = "onnx-community/vitpose-base-simple";
const model = await AutoModel.from_pretrained(model_id, {
  from_flax: true,
  from_onnx: true,
  device: "gpu",
  dtype: "fp32",
});
const processor = await AutoImageProcessor.from_pretrained(model_id);

import { createCanvas, createImageData } from "canvas";

// Vitpose keypoint edges (pairs of indices)
const VITPOSE_EDGES = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

// Corresponding RGB colors for each edge (in same order)
const VITPOSE_COLORS = [
  [255, 0, 0],
  [255, 85, 0],
  [255, 170, 0],
  [255, 255, 0],
  [170, 255, 0],
  [85, 255, 0],
  [0, 255, 0],
  [0, 255, 85],
  [0, 255, 170],
  [0, 255, 255],
  [0, 170, 255],
  [0, 85, 255],
  [0, 0, 255],
  [85, 0, 255],
  [170, 0, 255],
  [255, 0, 255],
  [255, 0, 170],
  [255, 0, 85],
];

async function saveAsImage(image, estimationResults) {
  const width = image.width;
  const height = image.height;
  const points = estimationResults.keypoints;

  // === 1. Create overlay image (pose + original image)
  const overlayCanvas = createCanvas(width, height);
  const overlayCtx = overlayCanvas.getContext("2d");

  const imageData = createImageData(image.rgba().data, width, height);
  overlayCtx.putImageData(imageData, 0, 0);

  drawPose(overlayCtx, points);

  const overlayOut = fs.createWriteStream(
    path.join(currentDir, "pose_overlay.png")
  );
  const overlayStream = overlayCanvas.createPNGStream();
  overlayStream.pipe(overlayOut);
  overlayOut.on("finish", () => console.log("Overlay PNG created."));

  // === 2. Create transparent pose-only image
  const transparentCanvas = createCanvas(width, height);
  const transparentCtx = transparentCanvas.getContext("2d");

  drawPose(transparentCtx, points);

  const transparentOut = fs.createWriteStream(
    path.join(currentDir, "pose_only.png")
  );
  const transparentStream = transparentCanvas.createPNGStream();
  transparentStream.pipe(transparentOut);
  transparentOut.on("finish", () =>
    console.log("Transparent pose PNG created.")
  );
}

function drawPose(ctx, points) {
  // Draw edges with their assigned RGB colors
  ctx.lineWidth = 4;
  VITPOSE_EDGES.forEach(([i, j], index) => {
    const [x1, y1] = points[i];
    const [x2, y2] = points[j];
    const [r, g, b] = VITPOSE_COLORS[index];
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });

  // Draw red keypoints
  ctx.fillStyle = "red";
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
  }
}

async function main() {
  // Load image and prepare inputs
  const url = path.join(currentDir, "input.webp");
  const image = await RawImage.read(url);
  const inputs = await processor(image);

  // Predict heatmaps
  const { heatmaps } = await model(inputs);

  // Post-process heatmaps to get keypoints and scores
  const boxes = [[[0, 0, image.width, image.height]]];
  const results = processor.post_process_pose_estimation(heatmaps, boxes)[0][0];
  await saveAsImage(image, results);
  console.log(results);
  console.log("Pose estimation completed and saved as pose.png");
}

main();
