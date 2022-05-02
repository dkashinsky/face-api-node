import * as tf from '@tensorflow/tfjs-node';

import { FaceDetection } from '../classes/FaceDetection';
import { extractFaceTensors, TNetInput } from '../core';
import { WithFaceDetection } from '../factories/WithFaceDetection';
import { isWithFaceLandmarks, WithFaceLandmarks } from '../factories/WithFaceLandmarks';

export async function extractAllFacesAndComputeResults<TSource extends WithFaceDetection<{}>, TResult>(
  parentResults: TSource[],
  input: TNetInput,
  computeResults: (faces: Array<tf.Tensor3D>) => Promise<TResult>,
  extractedFaces?: Array<tf.Tensor3D> | null,
  getRectForAlignment: (parentResult: WithFaceLandmarks<TSource, any>) => FaceDetection = ({ alignedRect }) => alignedRect
) {
  const faceBoxes = parentResults.map(parentResult =>
    isWithFaceLandmarks(parentResult)
      ? getRectForAlignment(parentResult)
      : parentResult.detection
  )
  const faces: Array<tf.Tensor3D> = extractedFaces || await extractFaceTensors(input, faceBoxes)

  const results = await computeResults(faces)

  faces.forEach(f => f instanceof tf.Tensor && f.dispose())

  return results
}

export async function extractSingleFaceAndComputeResult<TSource extends WithFaceDetection<{}>, TResult>(
  parentResult: TSource,
  input: TNetInput,
  computeResult: (face: tf.Tensor3D) => Promise<TResult>,
  extractedFaces?: Array<tf.Tensor3D> | null,
  getRectForAlignment?: (parentResult: WithFaceLandmarks<TSource, any>) => FaceDetection
) {
  return extractAllFacesAndComputeResults<TSource, TResult>(
    [parentResult],
    input,
    async faces => computeResult(faces[0]),
    extractedFaces,
    getRectForAlignment
  )
}
