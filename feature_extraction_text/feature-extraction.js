import dotenv from "dotenv";
dotenv.config();
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

async function generateEmbeddingWithModel(model, texts) {
  const embeddings = await client.featureExtraction({
    model,
    inputs: texts,
  });
  return embeddings;
}

function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function main() {
  const embeddings = await generateEmbeddingWithModel(
    "intfloat/multilingual-e5-large",
    ["query: Today is a beautiful day", "query: Today is a great day"]
  );
  const similarity = cosineSimilarity(embeddings[0], embeddings[1]);
  console.log("Cosine Similarity:", similarity);
}

main();
