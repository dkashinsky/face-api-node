import * as tf from '@tensorflow/tfjs-node';

import { toNetInput } from '../../../src';
import { AgeAndGenderPrediction } from '../../../src/ageGenderNet/types';
import { nodeTestEnv } from '../../env';
import { describeWithBackend, describeWithNets, expectAllTensorsReleased } from '../../utils';

function expectResultsAngry(result: AgeAndGenderPrediction) {
  expect(result.age).toBeGreaterThanOrEqual(36)
  expect(result.age).toBeLessThanOrEqual(42)
  expect(result.gender).toEqual('male')
  expect(result.genderProbability).toBeGreaterThanOrEqual(0.9)
}

function expectResultsSurprised(result: AgeAndGenderPrediction) {
  expect(result.age).toBeGreaterThanOrEqual(24)
  expect(result.age).toBeLessThanOrEqual(28)
  expect(result.gender).toEqual('female')
  expect(result.genderProbability).toBeGreaterThanOrEqual(0.8)
}

describeWithBackend('ageGenderNet', () => {

  let imgAngry: tf.Tensor3D
  let imgSurprised: tf.Tensor3D

  beforeAll(async () => {
    imgAngry = tf.node.decodeJpeg(await nodeTestEnv.loadImage('test/images/angry_cropped.jpg'))
    imgSurprised = tf.node.decodeJpeg(await nodeTestEnv.loadImage('test/images/surprised_cropped.jpg'))
  })

  describeWithNets('quantized weights', { withAgeGenderNet: { quantized: true } }, ({ ageGenderNet }) => {

    it('recognizes age and gender', async () => {
      const result = await ageGenderNet.predictAgeAndGender(imgAngry) as AgeAndGenderPrediction
      expectResultsAngry(result)
    })

  })

  describeWithNets('batch inputs', { withAgeGenderNet: { quantized: true } }, ({ ageGenderNet }) => {

    it('computes age and gender for batch of tf.Tensor3D', async () => {
      const inputs = [imgAngry, imgSurprised]

      const results = await ageGenderNet.predictAgeAndGender(inputs) as AgeAndGenderPrediction[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(2)

      const [resultAngry, resultSurprised] = results
      expectResultsAngry(resultAngry)
      expectResultsSurprised(resultSurprised)
    })

  })

  describeWithNets('no memory leaks', { withAgeGenderNet: { quantized: true } }, ({ ageGenderNet }) => {

    describe('forwardInput', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(async () => {
          const { age, gender } = await ageGenderNet.forwardInput(await toNetInput(imgAngry))
          age.dispose()
          gender.dispose()
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgAngry, imgAngry, imgAngry]

        await expectAllTensorsReleased(async () => {
          const { age, gender } = await ageGenderNet.forwardInput(await toNetInput(tensors))
          age.dispose()
          gender.dispose()
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgAngry.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          const { age, gender } = await ageGenderNet.forwardInput(await toNetInput(tensor))
          age.dispose()
          gender.dispose()
        })
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgAngry, imgAngry, imgAngry]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          const { age, gender } = await ageGenderNet.forwardInput(await toNetInput(tensors))
          age.dispose()
          gender.dispose()
        })

        tensors.forEach(t => t.dispose())
      })

    })

    describe('predictExpressions', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(async () => {
          await ageGenderNet.predictAgeAndGender(imgAngry)
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgAngry, imgAngry, imgAngry]

        await expectAllTensorsReleased(async () => {
          await ageGenderNet.predictAgeAndGender(tensors)
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgAngry.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          await ageGenderNet.predictAgeAndGender(tensor)
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgAngry, imgAngry, imgAngry]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          await ageGenderNet.predictAgeAndGender(tensors)
        })

        tensors.forEach(t => t.dispose())
      })

    })
  })

})

