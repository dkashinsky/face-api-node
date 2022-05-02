import { nodeTestEnv } from '../../env';

describe('fetchNetWeights', () => {

  it('fetches .weights file', async () => {
    const path = 'test/data/dummy.weights'
    const weights = await nodeTestEnv.loadNetWeights(path)

    expect(weights instanceof Float32Array).toBe(true)
  })

})
