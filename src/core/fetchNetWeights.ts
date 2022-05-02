import { readFile } from 'fs/promises';

export async function fetchNetWeights(filePath: string): Promise<Float32Array> {
  const fileBuffer = await readFile(filePath);
  return new Float32Array(fileBuffer);
}
