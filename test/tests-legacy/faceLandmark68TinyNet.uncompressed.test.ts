import * as tf from '@tensorflow/tfjs-node';
import { Point } from '../../src';
import { FaceLandmarks68 } from '../../src/classes/FaceLandmarks68';
import { nodeTestEnv } from '../env';
import { describeWithBackend, describeWithNets, expectPointClose } from '../utils';

describeWithBackend('faceLandmark68TinyNet, uncompressed', () => {

  let imgEl1: tf.Tensor3D
  let imgElRect: tf.Tensor3D
  let faceLandmarkPositions1: Point[]
  let faceLandmarkPositionsRect: Point[]

  beforeAll(async () => {
    imgEl1 = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'), 3)
    imgElRect = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face_rectangular.png'), 3)
    faceLandmarkPositions1 = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositions1Tiny.json')
    faceLandmarkPositionsRect = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositionsRectTiny.json')
  })

  afterAll(() => {
    imgEl1.dispose()
    imgElRect.dispose()
  })

  describeWithNets('uncompressed weights', { withFaceLandmark68TinyNet: { quantized: false } }, ({ faceLandmark68TinyNet }) => {

    it('computes face landmarks for squared input', async () => {
      const [height, width] = imgEl1.shape

      const result = await faceLandmark68TinyNet.detectLandmarks(imgEl1) as FaceLandmarks68
      expect(result.imageWidth).toEqual(width)
      expect(result.imageHeight).toEqual(height)
      expect(result.shift.x).toEqual(0)
      expect(result.shift.y).toEqual(0)
      result.positions.forEach((pt, i) => {
        const { x, y } = faceLandmarkPositions1[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

    it('computes face landmarks for rectangular input', async () => {
      const [height, width] = imgElRect.shape

      const result = await faceLandmark68TinyNet.detectLandmarks(imgElRect) as FaceLandmarks68
      expect(result.imageWidth).toEqual(width)
      expect(result.imageHeight).toEqual(height)
      expect(result.shift.x).toEqual(0)
      expect(result.shift.y).toEqual(0)
      result.positions.forEach((pt, i) => {
        const { x, y } = faceLandmarkPositionsRect[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

  })

})

