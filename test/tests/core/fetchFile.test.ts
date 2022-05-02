import { nodeTestEnv } from '../../env';

describe('fetchFile', () => {

  it('throws when file is not present', async () => {
    const path = '/does/not/exist'

    await expect(nodeTestEnv.loadJson(path)).rejects.toThrow()
  })

})
