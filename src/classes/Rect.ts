import { Box } from './Box';

export class Rect extends Box {
  constructor(x: number, y: number, width: number, height: number, allowNegativeDimensions: boolean = false) {
    super({ x, y, width, height }, allowNegativeDimensions)
  }
}
