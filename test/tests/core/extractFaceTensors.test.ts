import { extractFaceTensors, Rect, tf } from '../../../src';
import { describeWithBackend } from '../../utils';
import { nodeTestEnv } from '../../env';

describeWithBackend('extractFaceTensors', () => {

  let imgTensor: tf.Tensor3D

  beforeAll(async () => {
    imgTensor = tf.node.decodePng(await nodeTestEnv.loadImage('test/images/face1.png'), 3)
  })

  afterAll(async () => {
    imgTensor.dispose()
  })

  describe('extracts tensors', () => {

    it('single box', async () => {
      const rect = new Rect(0, 0, 50, 60)
      const tensors = await extractFaceTensors(imgTensor, [rect])

      expect(tensors.length).toEqual(1)
      expect(tensors[0].shape).toEqual([60, 50, 3])

      tensors.forEach(t => t.dispose())
    })

    it('multiple boxes', async () => {
      const rects = [
        new Rect(0, 0, 50, 60),
        new Rect(50, 50, 70, 80),
      ]
      const tensors = await extractFaceTensors(imgTensor, rects)

      expect(tensors.length).toEqual(2)
      expect(tensors[0].shape).toEqual([60, 50, 3])
      expect(tensors[1].shape).toEqual([80, 70, 3])

      tensors.forEach(t => t.dispose())
    })

  })

  describe('box out of image borders', () => {

    it('clips upper left corner', async () => {
      const rect = new Rect(-10, -10, 110, 110)
      const tensors = await extractFaceTensors(imgTensor, [rect])

      expect(tensors[0].shape).toEqual([100, 100, 3])

      tensors.forEach(t => t.dispose())
    })

    it('clips bottom right corner', async () => {
      const rect = new Rect(imgTensor.shape[1] - 100, imgTensor.shape[0] - 100, 110, 110)
      const tensors = await extractFaceTensors(imgTensor, [rect])

      expect(tensors[0].shape).toEqual([100, 100, 3])

      tensors.forEach(t => t.dispose())
    })

    it('clips upper left and bottom right corners', async () => {
      const rect = new Rect(-10, -10, imgTensor.shape[1] + 20, imgTensor.shape[0] + 20)
      const tensors = await extractFaceTensors(imgTensor, [rect])

      expect(tensors[0].shape).toEqual([imgTensor.shape[1], imgTensor.shape[0], 3])

      tensors.forEach(t => t.dispose())
    })

  })

})
