import * as tf from '@tensorflow/tfjs-node';

import { NetInput, toNetInput } from '../../../src';
import { nodeTestEnv } from '../../env';
import { expectAllTensorsReleased } from '../../utils';


describe('toNetInput', () => {

  let imgTensor: tf.Tensor3D

  beforeAll(async () => {
    imgTensor = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/white.png'))
  })

  afterAll(async () => {
    imgTensor.dispose()
  })

  describe('valid args', () => {

    it('from Tensor3D', async () => {
      const netInput = await toNetInput(imgTensor)
      expect(netInput instanceof NetInput).toBe(true)
      expect(netInput.batchSize).toEqual(1)
    })

    it('from Tensor4D', async () => {
      const img4d = imgTensor.expandDims<tf.Tensor4D>();
      const netInput = await toNetInput(img4d)
      expect(netInput instanceof NetInput).toBe(true)
      expect(netInput.batchSize).toEqual(1)
    })

    it('from mixed media array', async () => {
      const img4d = imgTensor.expandDims<tf.Tensor4D>();
      const netInput = await toNetInput([
        imgTensor,
        img4d,
      ])
      expect(netInput instanceof NetInput).toBe(true)
      expect(netInput.batchSize).toEqual(3)
    })

  })

  describe('invalid args', () => {
    it('undefined', async () => {
      let errorMessage
      try {
        await toNetInput(undefined as any)
      } catch (error) {
        errorMessage = error.message;
      }
      expect(errorMessage).toBe('toNetInput - expected media to be of type HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | tf.Tensor3D, or to be an element id')
    })

    it('empty array', async () => {
      let errorMessage
      try {
        await toNetInput([])
      } catch (error) {
        errorMessage = error.message;
      }
      expect(errorMessage).toBe('toNetInput - empty array passed as input')
    })

  })

  describe('no memory leaks', () => {

    it('constructor', async () => {
      const tensors = [imgTensor, imgTensor, imgTensor]
      const tensor4ds = tensors.map(t => t.expandDims<tf.Tensor4D>())

      await expectAllTensorsReleased(async () => {
        await toNetInput(imgTensor)
        await toNetInput([imgTensor, imgTensor, imgTensor])
        await toNetInput(tensors[0])
        await toNetInput(tensors)
        await toNetInput(tensor4ds[0])
        await toNetInput(tensor4ds)
      })

      tensor4ds.forEach(t => t.dispose())
    })

  })

})
