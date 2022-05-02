export type P<T, Pt> = {
  [Key in keyof T]: T[Key] extends Pt ? Pt : never;
};

type NumProps = P<FaceExpressions, number>;

export const FACE_EXPRESSION_LABELS: (keyof NumProps)[] = [
  'neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'
]

export class FaceExpressions {
  public neutral: number = 0
  public happy: number = 0
  public sad: number = 0
  public angry: number = 0
  public fearful: number = 0
  public disgusted: number = 0
  public surprised: number = 0

  constructor(probabilities: number[] | Float32Array) {
    if (probabilities.length !== 7) {
      throw new Error(`FaceExpressions.constructor - expected probabilities.length to be 7, have: ${probabilities.length}`)
    }

    FACE_EXPRESSION_LABELS.forEach((expression, idx) => {
      this[expression] = probabilities[idx] as any
    })
  }

  asSortedArray() {
    return FACE_EXPRESSION_LABELS
      .map(expression => ({ expression, probability: this[expression] as number }))
      .sort((e0, e1) => e1.probability - e0.probability)
  }
}
