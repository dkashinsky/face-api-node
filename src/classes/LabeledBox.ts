import { isValidNumber } from '../utils';
import { IBoundingBox, IRect } from "./types";
import { Box } from './Box';

export class LabeledBox extends Box {

  public static assertIsValidLabeledBox(box: any, callee: string) {
    Box.assertIsValidBox(box, callee)

    if (!isValidNumber(box.label)) {
      throw new Error(`${callee} - expected property label (${box.label}) to be a number`)
    }
  }

  private _label: number

  constructor(box: IBoundingBox | IRect | any, label: number) {
    super(box)
    this._label = label
  }

  public get label(): number { return this._label }

}
