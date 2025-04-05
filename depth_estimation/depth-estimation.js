import { pipeline } from "@huggingface/transformers";
import path from "path";
import { fileURLToPath } from "url";

const currentDir = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const depthEstimation = await pipeline(
    "depth-estimation",
    "onnx-community/depth-anything-v2-small"
  );
  const url = path.join(currentDir, "input.webp");
  const { depth } = await depthEstimation(url);
  depth.save(path.join(currentDir, "output.png"));
  console.log("Depth map saved as depth.png");
}

main();
