import * as tf from '@tensorflow/tfjs-node';

import { IDimensions, FaceLandmarks68, utils, NetInput, Point, toNetInput } from '../../../src';
import { nodeTestEnv } from '../../env';
import {
  describeWithBackend,
  describeWithNets,
  expectAllTensorsReleased,
  expectMaxDelta,
  expectPointClose,
} from '../../utils';

function getInputDims (input: tf.Tensor): IDimensions {
  if (input instanceof tf.Tensor) {
    const [height, width] = input.shape.slice(utils.isTensor3D(input) ? 0 : 1)
    return { width, height }
  }
  return input
}

describeWithBackend('faceLandmark68Net', () => {

  let imgEl1: tf.Tensor3D
  let imgEl2: tf.Tensor3D
  let imgElRect: tf.Tensor3D
  let faceLandmarkPositions1: Point[]
  let faceLandmarkPositions2: Point[]
  let faceLandmarkPositionsRect: Point[]

  beforeAll(async () => {
    imgEl1 = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'), 3)
    imgEl2 = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face2.png'), 3)
    imgElRect = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face_rectangular.png'), 3)
    faceLandmarkPositions1 = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositions1.json')
    faceLandmarkPositions2 = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositions2.json')
    faceLandmarkPositionsRect = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositionsRect.json')
  })

  describeWithNets('quantized weights', { withFaceLandmark68Net: { quantized: true } }, ({ faceLandmark68Net }) => {

    it('computes face landmarks for squared input', async () => {
      const [height, width] = imgEl1.shape

      const result = await faceLandmark68Net.detectLandmarks(imgEl1) as FaceLandmarks68
      expect(result.imageWidth).toEqual(width)
      expect(result.imageHeight).toEqual(height)
      expect(result.shift.x).toEqual(0)
      expect(result.shift.y).toEqual(0)
      result.positions.forEach((pt, i) => {
        const { x, y } = faceLandmarkPositions1[i]
        expectPointClose(pt, { x, y }, 3)
      })
    })

    it('computes face landmarks for rectangular input', async () => {
      const [height, width] = imgElRect.shape

      const result = await faceLandmark68Net.detectLandmarks(imgElRect) as FaceLandmarks68
      expect(result.imageWidth).toEqual(width)
      expect(result.imageHeight).toEqual(height)
      expect(result.shift.x).toEqual(0)
      expect(result.shift.y).toEqual(0)
      result.positions.forEach((pt, i) => {
        const { x, y } = faceLandmarkPositionsRect[i]
        expectPointClose(pt, { x, y }, 6)
      })
    })

  })

  describeWithNets('batch inputs', { withFaceLandmark68Net: { quantized: true } }, ({ faceLandmark68Net }) => {

    it('computes face landmarks for batch of tf.Tensor3D', async () => {
      const inputs = [imgEl1, imgEl2, imgElRect]

      const faceLandmarkPositions = [
        faceLandmarkPositions1,
        faceLandmarkPositions2,
        faceLandmarkPositionsRect
      ]

      const results = await faceLandmark68Net.detectLandmarks(inputs) as FaceLandmarks68[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(3)
      results.forEach((result, batchIdx) => {
        const { width, height } = getInputDims(inputs[batchIdx])
        expect(result.imageWidth).toEqual(width)
        expect(result.imageHeight).toEqual(height)
        expect(result.shift.x).toEqual(0)
        expect(result.shift.y).toEqual(0)
        result.positions.forEach(({ x, y }, i) => {
          expectMaxDelta(x, faceLandmarkPositions[batchIdx][i].x, 3)
          expectMaxDelta(y, faceLandmarkPositions[batchIdx][i].y, 3)
        })
      })
    })

  })

  describeWithNets('no memory leaks', { withFaceLandmark68Net: { quantized: true } }, ({ faceLandmark68Net }) => {

    describe('forwardInput', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(() => {
          const netInput = new NetInput([imgEl1])
          const outTensor = faceLandmark68Net.forwardInput(netInput)
          outTensor.dispose()
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]

        await expectAllTensorsReleased(async () => {
          const netInput = new NetInput(tensors)
          const outTensor = faceLandmark68Net.forwardInput(netInput)
          outTensor.dispose()
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgEl1.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          const outTensor = faceLandmark68Net.forwardInput(await toNetInput(tensor))
          outTensor.dispose()
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          const outTensor = faceLandmark68Net.forwardInput(await toNetInput(tensors))
          outTensor.dispose()
        })

        tensors.forEach(t => t.dispose())
      })

    })

    describe('detectLandmarks', () => {

      it('single tf.Tensor3D', async () => {
        await expectAllTensorsReleased(async () => {
          await faceLandmark68Net.detectLandmarks(imgEl1)
        })
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]

        await expectAllTensorsReleased(async () => {
          await faceLandmark68Net.detectLandmarks(tensors)
        })
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => imgEl1.expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          await faceLandmark68Net.detectLandmarks(tensor)
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]
          .map(el => tf.tidy(() => el.expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          await faceLandmark68Net.detectLandmarks(tensors)
        })

        tensors.forEach(t => t.dispose())
      })

    })
  })

})

