import { fetchNetWeights } from '../../../src';

describe('fetchNetWeights', () => {

  it('fetches .weights file', async () => {
    const path = 'base/test/data/dummy.weights'
    const weights = await fetchNetWeights(path)
    expect(weights instanceof Float32Array).toBe(true)
  })

})
