import { nodeTestEnv } from '../../env';

describe('fetchFile', () => {

  it('throws when file is not present', async () => {
    const path = '/does/not/exist'

    let err = ''
    try {
      await nodeTestEnv.loadImage(path)
    } catch (e) {
      err = e.toString()
    }

    expect(err).toBeTruthy()
  })

})
