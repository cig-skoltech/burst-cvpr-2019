import json

import cv2
import matplotlib.pyplot as plt
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def loadmat(filename):
    import scipy.io as spio
    import scipy
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    link: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def plot_burst(burst):
    n = burst.shape[0]
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, n + 1):
        plt.subplot(n // 2, n // 2, i)
        plt.imshow(burst[i - 1])
    plt.show()


def realistic_noise(img, a, b):
    assert (img.min() >= 0 and img.max() <= 1), 'image range should be between 0 and 1'
    y = np.random.normal(loc=img, scale=a * img + b ** 2)
    if img.dtype == np.uint8:
        y = y.astype(np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        y = y.astype(img.dtype).clip(0, 1)
    return y


def srgb_to_linrgb(img):
    """ Convert sRGB color space to linRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
    assert img.dtype in [np.float32, np.float64]
    img = img.copy()
    mask = img <= 0.04045
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    img[mask] = img[mask] / 12.92
    return img


def linrgb_to_srgb(img):
    """ Convert linRGB color space to sRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
    assert img.dtype in [np.float32, np.float64]
    img = img.copy()
    realistic_noise
    mask = img <= 0.0031308
    img[~mask] = (img[~mask] ** (1 / 2.4)) * (1.055) - 0.055
    img[mask] = img[mask] * 12.92
    return img


def calculate_affine_matrices(burst):
    warp_matrices = np.zeros((burst.shape[0], 2, 3))
    warp_matrices[-1] = np.eye(2, 3)  # identity for reference frame
    for i, b in enumerate(burst[:-1]):
        warp_matrix = calculate_ECC((burst[-1] / 255).astype(np.float32), (b / 255).astype(np.float32), 2)
        if warp_matrix is None:
            return None

        warp_matrices[i] = warp_matrix
    return warp_matrices


def calculate_ECC(img_ref, img, nol=4):
    img_ref = bayer_to_gray(img_ref)
    img = bayer_to_gray(img)
    # ECC params
    init_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    n_iters = 3000
    e_thresh = 1e-6
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, e_thresh)

    warp = init_warp

    # construct grayscale pyramid
    gray1_pyr = [img_ref]
    gray2_pyr = [img]

    for level in range(nol - 1):
        gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1 / 2, fy=1 / 2,
                                       interpolation=cv2.INTER_AREA))
        gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1 / 2, fy=1 / 2,
                                       interpolation=cv2.INTER_AREA))

    # run pyramid ECC
    error_cnt = 0
    for level in range(nol):
        try:
            cc, warp_ = cv2.findTransformECC(gray1_pyr[level], gray2_pyr[level],
                                             warp, warp_mode, criteria)
            warp = warp_
            if level != nol - 1:  # scale up for the next pyramid level
                warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
        except Exception as e:
            error_cnt += 1
            pass

    if error_cnt == nol:
        return None
    else:
        warp[:, 2] *= 2
        return warp


def bayer_to_gray(img):
    g1 = img[1::2, ::2, 1]
    g2 = img[::2, 1::2, 1]
    g = (g1 + g2) / 2
    r = img[::2, ::2, 0]
    b = img[1::2, 1::2, 2]
    y = 0.2125 * r + 0.7154 * g + 0.0721 * b
    return y


def plot_batch_burst(burst):
    batch, n, _, _, _ = burst.shape
    burst = burst.reshape(-1, burst.shape[2], burst.shape[3], burst.shape[4])
    fig = plt.figure(figsize=(15, 3.9))
    fig.subplots_adjust(wspace=0.1, hspace=0.01)
    for i in range(1, batch * n + 1):
        plt.subplot(batch, n, i)
        plt.imshow(burst[i - 1].cpu().data.permute(1, 2, 0).numpy().astype(np.uint8))
        plt.axis('off')
    # plt.tight_layout()
    plt.show()
