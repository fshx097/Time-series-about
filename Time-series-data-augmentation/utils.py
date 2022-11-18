import random
import tsaug
import np
'''
the time-series's form is (len_sequnece,channaels)
'''
def add_nosie(x):
    y = np.array(x)
    for i in range(y.shape[-1]):
        y[:,i] = tsaug.AddNoise(scale=0.05).augment(y[:,i])
    return y

#反转序列
def reverse(x):
    y = np.array(x)
    for i in range(y.shape[-1]):
        y[:,i] = tsaug.Reverse().augment(y[:,i])
    return y


# 池化
def pool(x, kind, size):
    y = np.array(x)
    # 目前不知道tsaug的运作机制
    if kind == 'min':
        for i in range(y.shape[-1]):
            y[:, i] = tsaug.Pool(size=size, kind='min').augment(y[:, i])

    elif kind == 'max':
        # 普遍前向填充
        q = y[0, :].reshape(1, -1).repeat(size - 1, axis=0)
        y1 = np.concatenate((q, y))
        for i in range(y.shape[0]):
            y[i, :] = np.max(y1[i:i + size, :], axis=0)

    elif kind == 'ave':
        # 普遍前向填充
        q = y[0, :].reshape(1, -1).repeat(size - 1, axis=0)
        y1 = np.concatenate((q, y))
        for i in range(y.shape[0]):
            y[i, :] = np.sum(y1[i:i + size, :], axis=0) / size

    return y


#上采样
def up_sample(x,size):
    y = np.array(x).transpose()
    y = tsaug.Resize(size, repeats=1, prob=1, seed=None).augment(y)
    y = y.transpose()
    return y

#中心对称
def center_rotate(x):
    y = np.array(x)
    y = y[::-1]
    return y

#中心镜面翻转
def center_flip(x,center):
    '''
    x:(len_seq,channels)
    center is the number of center-point in seq
    '''
    y = np.array(x)
    for i in range(x.shape[1]):
        y[:,i] = y[:,i] - 2 * (y[:,i] - y[center,i])
    return y

#沿x轴反转
def flip(x):
    '''
    x:(len_seq,channels)
    center is the number of center-point in seq
    '''
    y = np.array(x)
    y = -y
    return y