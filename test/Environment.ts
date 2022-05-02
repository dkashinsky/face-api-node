import { NeuralNetwork } from '../src';

export type TestEnv = {
  loadNetWeights: (uri: string) => Promise<Float32Array>,
  loadImage: (uri: string) => Promise<Buffer>,
  loadJson: <T> (uri: string) => Promise<T>
  initNet: <TNet extends NeuralNetwork<any>>(
    net: TNet,
    uncompressedFilename?: string | boolean,
    isUnusedModel?: boolean
  ) => any
}
