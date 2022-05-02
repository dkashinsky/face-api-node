import * as tf from '@tensorflow/tfjs-core';

import { NetInput } from './NetInput';

export type TResolvedNetInput = tf.Tensor3D | tf.Tensor4D

export type TNetInputArg = TResolvedNetInput

export type TNetInput = TNetInputArg | Array<TNetInputArg> | NetInput
