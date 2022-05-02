import * as tf from '@tensorflow/tfjs-node';

import { NetInput, toNetInput } from '../../../src';
import { FaceExpressions } from '../../../src/faceExpressionNet/FaceExpressions';
import { nodeTestEnv } from '../../env';
import { describeWithBackend, describeWithNets, expectAllTensorsReleased } from '../../utils';

describeWithBackend('faceExpressionNet', () => {

  let imgElAngry: tf.Tensor3D
  let imgElSurprised: tf.Tensor3D

  beforeAll(async () => {
    imgElAngry = tf.node.decodeJpeg(await nodeTestEnv.loadImage('test/images/angry_cropped.jpg'))
    imgElSurprised = tf.node.decodeJpeg(await nodeTestEnv.loadImage('test/images/surprised_cropped.jpg'))
  })

  afterAll(() => {
    imgElAngry.dispose()
    imgElSurprised.dispose()
  })

  describeWithNets('quantized weights', { withFaceExpressionNet: { quantized: true } }, ({ faceExpressionNet }) => {

    it('recognizes facial expressions', async () => {
      const result = await faceExpressionNet.predictExpressions(imgElAngry) as FaceExpressions
      expect(result instanceof FaceExpressions).toBe(true)
      expect(result.angry).toBeGreaterThan(0.95)
    })

  })

  describeWithNets('batch inputs', { withFaceExpressionNet: { quantized: true } }, ({ faceExpressionNet }) => {

    it('computes face expressions for batch of tf.Tensor3D', async () => {
      const inputs = [imgElAngry, imgElSurprised]

      const results = await faceExpressionNet.predictExpressions(inputs) as FaceExpressions[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(2)

      const [resultAngry, resultSurprised] = results
      expect(resultAngry instanceof FaceExpressions).toBe(true)
      expect(resultSurprised instanceof FaceExpressions).toBe(true)
      expect(resultAngry.angry).toBeGreaterThan(0.95)
      expect(resultSurprised.surprised).toBeGreaterThan(0.95)
    })

  })

  describeWithNets('no memory leaks', { withFaceExpressionNet: { quantized: true } }, ({ faceExpressionNet }) => {

    describe('forwardInput', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(async () => {
          const outTensor = faceExpressionNet.forwardInput(await toNetInput(imgElAngry))
          outTensor.dispose()
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgElAngry, imgElAngry, imgElAngry]

        await expectAllTensorsReleased(async () => {
          const outTensor = faceExpressionNet.forwardInput(await toNetInput(tensors))
          outTensor.dispose()
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgElAngry.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          const outTensor = faceExpressionNet.forwardInput(await toNetInput(tensor))
          outTensor.dispose()
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgElAngry, imgElAngry, imgElAngry]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          const outTensor = faceExpressionNet.forwardInput(await toNetInput(tensors))
          outTensor.dispose()
        })

        tensors.forEach(t => t.dispose())
      })

    })

    describe('predictExpressions', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(async () => {
          await faceExpressionNet.predictExpressions(imgElAngry)
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgElAngry, imgElAngry, imgElAngry]

        await expectAllTensorsReleased(async () => {
          await faceExpressionNet.predictExpressions(tensors)
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgElAngry.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          await faceExpressionNet.predictExpressions(tensor)
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgElAngry, imgElAngry, imgElAngry]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          await faceExpressionNet.predictExpressions(tensors)
        })

        tensors.forEach(t => t.dispose())
      })

    })
  })

})

