import { readFile } from 'fs/promises';

export async function fetchImage(filePath: string): Promise<Buffer> {
  return await readFile(filePath);
}

export const fetchFile = fetchImage;
