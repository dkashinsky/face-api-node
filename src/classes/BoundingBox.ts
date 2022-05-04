import { Box } from './Box';

export class BoundingBox extends Box {
  constructor(left: number, top: number, right: number, bottom: number, allowNegativeDimensions: boolean = false) {
    super({ left, top, right, bottom }, allowNegativeDimensions)
  }
}
