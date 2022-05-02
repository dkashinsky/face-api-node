import { nodeTestEnv } from '../../env';

describe('fetchJson', () => {

  it('fetches json', async () => {
    const filePath = 'test/data/boxes.json'
    expect(async () => await nodeTestEnv.loadJson(filePath)).not.toThrow()
  })

})
