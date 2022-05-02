import { readFile } from 'fs/promises';

export async function fetchJson<T>(filePath: string): Promise<T> {
  const content = await readFile(filePath, 'utf8');
  return JSON.parse(content);
}
