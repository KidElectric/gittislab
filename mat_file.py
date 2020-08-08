def load(path):
    from scipy.io import loadmat
    print('Loading %s...%s ...' % (str(path)[:20],str(path)[-30:]))
    dat=loadmat(path)
    print('\tFinished')
    return dat
