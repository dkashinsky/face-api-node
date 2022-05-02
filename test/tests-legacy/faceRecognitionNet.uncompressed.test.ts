import * as tf from '@tensorflow/tfjs-node';
import { euclideanDistance } from '../../src';
import { nodeTestEnv } from '../env';
import { describeWithBackend, describeWithNets } from '../utils';

describeWithBackend('faceRecognitionNet, uncompressed', () => {

  let imgEl1: tf.Tensor3D
  let imgElRect: tf.Tensor3D
  let faceDescriptor1: number[]
  let faceDescriptorRect: number[]

  beforeAll(async () => {
    imgEl1 = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'))
    imgElRect = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face_rectangular.png'))
    faceDescriptor1 = await nodeTestEnv.loadJson<number[]>('test/data/faceDescriptor1.json')
    faceDescriptorRect = await nodeTestEnv.loadJson<number[]>('test/data/faceDescriptorRect.json')
  })

  describeWithNets('uncompressed weights', { withFaceRecognitionNet: { quantized: false } }, ({ faceRecognitionNet }) => {

    it('computes face descriptor for squared input', async () => {
      const result = await faceRecognitionNet.computeFaceDescriptor(imgEl1) as Float32Array
      expect(result.length).toEqual(128)
      expect(euclideanDistance(result, faceDescriptor1)).toBeLessThan(0.1)
    })

    it('computes face descriptor for rectangular input', async () => {
      const result = await faceRecognitionNet.computeFaceDescriptor(imgElRect) as Float32Array
      expect(result.length).toEqual(128)
      expect(euclideanDistance(result, faceDescriptorRect)).toBeLessThan(0.1)
    })

  })
})
