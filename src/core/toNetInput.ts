import { isTensor3D, isTensor4D } from '../utils';
import { NetInput } from './NetInput';
import { TNetInput } from './types';

/**
 * Validates the input to make sure, they are valid net inputs and awaits all media elements
 * to be finished loading.
 *
 * @param input The input, which can be a media element or an array of different media elements.
 * @returns A NetInput instance, which can be passed into one of the neural networks.
 */
export async function toNetInput(inputs: TNetInput): Promise<NetInput> {
  if (inputs instanceof NetInput) {
    return inputs
  }

  let inputArgArray = Array.isArray(inputs)
      ? inputs
      : [inputs]

  if (!inputArgArray.length) {
    throw new Error('toNetInput - empty array passed as input')
  }

  const getIdxHint = (idx: number) => Array.isArray(inputs) ? ` at input index ${idx}:` : ''

  const tensors = inputArgArray.map((input, i) => {
    if (!isTensor3D(input) && !isTensor4D(input)) {
      throw new Error(`toNetInput -${getIdxHint(i)} expected to be of type tf.Tensor3D or tf.Tensor4D`)
    }

    if (isTensor4D(input)) {
      // if tf.Tensor4D is passed in the input array, the batch size has to be 1
      const batchSize = input.shape[0]
      if (batchSize !== 1) {
        throw new Error(`toNetInput -${getIdxHint(i)} tf.Tensor4D with batchSize ${batchSize} passed, but not supported in input array`)
      }
    }

    return input;
  })

  return new NetInput(tensors, Array.isArray(inputs))
}
