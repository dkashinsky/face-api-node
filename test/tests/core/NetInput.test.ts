import * as tf from '@tensorflow/tfjs-node';

import { NetInput } from '../../../src';
import { nodeTestEnv } from '../../env';
import { expectAllTensorsReleased, fakeTensor3d } from '../../utils';

describe('NetInput', () => {

  let imgTensor: tf.Tensor3D

  beforeAll(async () => {
    imgTensor = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'))
  })

  afterAll(async () => {
    imgTensor.dispose()
  })

  describe('toBatchTensor', () => {

    it('Tensor3D, batchSize === 1', () => tf.tidy(() => {
      const netInput = new NetInput([imgTensor])
      const batchTensor = netInput.toBatchTensor(100)
      expect(batchTensor.shape).toEqual([1, 100, 100, 3])
    }))

    it('Tensor3D, batchSize === 4', () => tf.tidy(() => {
      const netInput = new NetInput([
        imgTensor,
        imgTensor,
        imgTensor,
        imgTensor
      ])
      const batchTensor = netInput.toBatchTensor(100)
      expect(batchTensor.shape).toEqual([4, 100, 100, 3])
    }))

  })

  describe('no memory leaks', () => {

    it('constructor', async () => {
      const tensors = [fakeTensor3d(), fakeTensor3d(), fakeTensor3d()]

      await expectAllTensorsReleased(() => {
        new NetInput([imgTensor])
        new NetInput([imgTensor, imgTensor, imgTensor])
        new NetInput([tensors[0]])
        new NetInput(tensors)
      })

      tensors.forEach(t => t.dispose())
    })

    describe('toBatchTensor', () => {

      it('single image element', async () => {
        await expectAllTensorsReleased(() => {
          const batchTensor = new NetInput([imgTensor]).toBatchTensor(100, false)
          batchTensor.dispose()
        })
      })

      it('multiple image elements', async () => {
        await expectAllTensorsReleased(() => {
          const batchTensor = new NetInput([imgTensor, imgTensor, imgTensor]).toBatchTensor(100, false)
          batchTensor.dispose()
        })
      })

    })

  })

})
