import * as tf from '@tensorflow/tfjs-node';
import { FaceLandmarks68, Point } from '../../src';
import { nodeTestEnv } from '../env';
import { describeWithBackend, describeWithNets, expectPointClose } from '../utils';

describeWithBackend('faceLandmark68Net, uncompressed', () => {

  let imgEl1: tf.Tensor3D
  let imgElRect: tf.Tensor3D
  let faceLandmarkPositions1: Point[]
  let faceLandmarkPositionsRect: Point[]

  beforeAll(async () => {
    imgEl1 = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'), 3)
    imgElRect = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face_rectangular.png'), 3)
    faceLandmarkPositions1 = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositions1.json')
    faceLandmarkPositionsRect = await nodeTestEnv.loadJson<Point[]>('test/data/faceLandmarkPositionsRect.json')
  })

  afterAll(() => {
    imgEl1.dispose()
    imgElRect.dispose()
  })

  describeWithNets('uncompressed weights', { withFaceLandmark68Net: { quantized: false } }, ({ faceLandmark68Net }) => {

    it('computes face landmarks for squared input', async () => {
      const [height, width] = imgEl1.shape

      const result = await faceLandmark68Net.detectLandmarks(imgEl1) as FaceLandmarks68
      expect(result.imageWidth).toEqual(width)
      expect(result.imageHeight).toEqual(height)
      expect(result.shift.x).toEqual(0)
      expect(result.shift.y).toEqual(0)
      result.positions.forEach((pt, i) => {
        const { x, y } = faceLandmarkPositions1[i]
        expectPointClose(pt, { x, y }, 1)
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
        expectPointClose(pt, { x, y }, 5)
      })
    })

  })

})

