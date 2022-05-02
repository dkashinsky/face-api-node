import * as path from 'path';

import { fetchFile, fetchJson, fetchNetWeights, NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

async function loadNetWeights(uri: string): Promise<Float32Array> {
  return await fetchNetWeights(path.resolve(__dirname, '../', uri))
}

async function loadImage(uri: string): Promise<Buffer> {
  return await fetchFile(path.resolve(__dirname, '../', uri))
}

async function loadJson<T>(uri: string): Promise<T> {
  return await fetchJson(path.resolve(__dirname, '../', uri))
}

export async function initNet<TNet extends NeuralNetwork<any>>(net: TNet) {
  await net.loadFromDisk(path.resolve(__dirname, '../weights'))
}

export const nodeTestEnv: TestEnv = {
  loadNetWeights,
  loadImage,
  loadJson,
  initNet,
}
