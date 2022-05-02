import { nodeTestEnv } from '../../env';

describe('fetchJson', () => {

  it('fetches json', async () => {
    const filePath = 'test/data/boxes.json'

    await expect(nodeTestEnv.loadJson(filePath)).resolves.not.toThrow()
  })

})
