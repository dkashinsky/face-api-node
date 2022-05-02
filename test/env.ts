import { readFile } from 'fs/promises';
import * as path from 'path';

import { NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

async function loadImage(uri: string): Promise<Buffer> {
  return await readFile(path.resolve(__dirname, '../', uri))
}

async function loadJson<T>(uri: string): Promise<T> {
  const content = await readFile(path.resolve(__dirname, '../', uri), 'utf8')
  return JSON.parse(content)
}

export async function initNet<TNet extends NeuralNetwork<any>>(net: TNet) {
  await net.loadFromDisk(path.resolve(__dirname, '../weights'))
}

export const nodeTestEnv: TestEnv = {
  loadImage,
  loadJson,
  initNet,
}
