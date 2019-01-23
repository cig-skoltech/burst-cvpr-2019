import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from subprocess import call, check_output
from subprocess import Popen, PIPE
import cv2

plt.switch_backend('agg')


def worker_init_fn(pid):
    np.random.seed(42 + pid)
    torch.cuda.manual_seed(42 + pid)
    torch.manual_seed(42 + pid)


def plot_batch_burst(burst):
    batch, n, _, _, _ = burst.shape
    burst = burst.reshape(-1, burst.shape[2], burst.shape[3], burst.shape[4])
    fig = plt.figure(figsize=(8, 2))
    fig.subplots_adjust(wspace=0, hspace=0.1)
    for i in range(1, batch * n + 1):
        plt.subplot(batch, n, i)
        plt.imshow(burst[i - 1].cpu().data.permute(1, 2, 0).numpy().astype(np.uint8))
        plt.axis('off')
    # plt.tight_layout()
    return fig


def calc_psnr(prediction, target):
    # Calculate PSNR
    # Data have to be in range (0, 1)
    mse = torch.pow(prediction - target, 2).reshape(target.shape[0], -1)
    mse = mse.mean(dim=1)
    psnrs = 10 * torch.log10(1 / mse)
    psnr_list = list(psnrs.cpu().numpy())
    return psnr_list

def im2Tensor(img, dtype=th.FloatTensor):
    assert (isinstance(img, np.ndarray) and img.ndim in (2, 3, 4)), "A numpy " \
                                                                    "nd array of dimensions 2, 3, or 4 is expected."

    if img.ndim == 2:
        return th.from_numpy(img).unsqueeze_(0).unsqueeze_(0).type(dtype)
    elif img.ndim == 3:
        return th.from_numpy(img.transpose(2, 0, 1)).unsqueeze_(0).type(dtype)
    else:
        return th.from_numpy(img.transpose((3, 2, 0, 1))).type(dtype)


def tensor2Im(img, dtype=np.float32):
    assert (isinstance(img, th.Tensor) and img.ndimension() == 4), "A 4D " \
                                                                   "torch.Tensor is expected."
    fshape = (0, 2, 3, 1)

    return img.numpy().transpose(fshape).astype(dtype)


def loadmat(filename):
    import scipy.io as spio
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

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    return _check_keys(data)


def rescale_to_255f(img):
    dtype = img.dtype
    if dtype == np.uint16:
        img = img / 2**16 * 2**8
    elif dtype == np.uint8:
        img = img.astype(np.float32)
    return img

def check_pattern(img_path):
    p = Popen(['dcraw','-i','-v',img_path], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    output = str(output)
    cfa = output.split('Filter pattern:')[1][1:5]
    if 'RGGB' in cfa or 'Fujifilm' in output:
        rollx , rolly = 0, 0
    elif 'GBRG' in cfa:
        rollx , rolly = 0, 1
    elif 'GRBG' in cfa:
        rollx , rolly = 1, 0
    elif 'BGGR' in cfa:
        rollx , rolly = 1, 1
    else:
        raise NotImplementedError
    return rollx , rolly


def linrgb_to_srgb(img):
    """ Convert linRGB color space to sRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
    assert img.dtype in [np.float32, np.float64] 
    img = img.copy()
    mask = img < 0.0031308
    img[~mask] = (img[~mask]**(1/2.4))*(1.055) - 0.055
    img[mask] = img[mask] * 12.92
    return img


def calculate_affine_matrices(burst):
    r""" Calculate affine matrices for frames inside a burst """
    warp_matrices = np.zeros((burst.shape[0],2,3))
    warp_matrices[-1] = np.eye(2,3) # identity for reference frame
    for i, b in enumerate(burst[:-1]):
        warp_matrix = calculate_ECC((burst[-1]/255).astype(np.float32),(b/255).astype(np.float32), 1) 
        if warp_matrix is None:
            wapr_matrix = np.zeros((2,3))+np.inf

        warp_matrices[i] = warp_matrix
    return warp_matrices

def calculate_ECC(img_ref, img, nol=4):
    img_ref = bayer_to_gray(img_ref)
    img = bayer_to_gray(img)
    # ECC params
    init_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    n_iters = 5000
    e_thresh = 1e-6
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, e_thresh)

    warp = init_warp

    # construct grayscale pyramid
    gray1_pyr = [img_ref]
    gray2_pyr = [img]

    for level in range(nol-1):
        gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1/2, fy=1/2,
                                       interpolation=cv2.INTER_AREA))
        gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1/2, fy=1/2,
                                       interpolation=cv2.INTER_AREA))

    # run pyramid ECC
    error_cnt = 0
    for level in range(nol):
        try:
            cc, warp_ = cv2.findTransformECC(gray1_pyr[level], gray2_pyr[level],
                                            warp, warp_mode, criteria)
            warp = warp_
            if level != nol-1:  # scale up for the next pyramid level
                warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
        except Exception as e:
            print(level, e)
            error_cnt += 1
            pass

    if error_cnt == nol:
        return None
    else:
        warp[:,2] *= 2  # mosaicked image is downsampled by a factor of 2 during warp estimation,
                        # therefore the translation vector is multiplied by 2 at the end
                        # to account for the downsampling
        return warp

def burst_warp(burst):
    r""" Warp frames inside a burst """
    aligned_images = []
    masks = []
    w = calculate_affine_matrices(burst)
    for i, b in enumerate(burst):
        im = cv2.warpAffine(b, w[i], (b.shape[0],b.shape[1]),
                           flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        mask = im > 1
        im = cv2.warpAffine(b, w[i], (b.shape[0],b.shape[1]),
                           flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
        masks.append(mask.astype(np.uint16))
        aligned_images.append(im)
    return np.array(aligned_images),np.array(masks)

def calc_img(aligned_images, masks):
    return (masks*aligned_images).sum(axis=0) / masks.sum(axis=0)

def bayer_to_gray(img):
    r""" Convert bayer image to gray scale by downsampling
         the periodic pattern.
     """
    g1 = img[1::2,::2,1]
    g2 = img[::2,1::2,1]
    g = (g1+g2)/2
    r = img[::2,::2,0]
    b = img[1::2,1::2,2]
    y = 0.2125*r + 0.7154*g + 0.0721*b
    return y

def compress(cfa_img):
    r""" Compress a mosaicked image. The sequence according to channels is: R, G1, G2, B

         Input:
             x with shape [b, 3, H, W]
         Return:
             compressed_image of shape [b, 4, H/2, W/2]
    """
    cfa_im_comp = np.zeros((int(cfa_img.shape[0]/2),int(cfa_img.shape[1]/2),
                            cfa_img.shape[2]+1))
    cfa_im_comp[...,0] = cfa_img[::2,::2,0] # R
    cfa_im_comp[...,1] = cfa_img[::2,1::2,1] # G
    cfa_im_comp[...,2] = cfa_img[1::2,::2,1] # G
    cfa_im_comp[...,3] = cfa_img[1::2,1::2,2] # B
    return cfa_im_comp

def decompress_burst(cfa_img):
    r""" Decompress an image.

         Input:
            x with shape [b, 4, H/2, W/2]
         Return:
            decompressed_image of shape [b, 3, H, W]
     """
    cfa_im_comp = np.zeros((cfa_img.shape[0],int(cfa_img.shape[1]*2),int(cfa_img.shape[2]*2),
                            cfa_img.shape[3]-1))
    cfa_im_comp[:,::2,::2,0] = cfa_img[...,0] # R
    cfa_im_comp[:,::2,1::2,1] = cfa_img[...,1] # G
    cfa_im_comp[:,1::2,::2,1] = cfa_img[...,2] # G
    cfa_im_comp[:,1::2,1::2,2] = cfa_img[...,3] # B
    return cfa_im_comp

def raw_to_bayer(cfa_img):
    bayer_img = np.zeros((cfa_img.shape[0], cfa_img.shape[1],3),dtype=cfa_img.dtype)
    bayer_img[::2,::2,0] = cfa_img[::2,::2] # R
    bayer_img[::2,1::2,1] = cfa_img[::2,1::2] # G
    bayer_img[1::2,::2,1] = cfa_img[1::2,::2] # G
    bayer_img[1::2,1::2,2] = cfa_img[1::2,1::2] # B
    return bayer_img


def bilinear_interpol_conv(mosaick_img):
    r""" Demosaick image using bilinear interpolation"""
    assert mosaick_img.max() <= 1 and mosaick_img.min() >= 0
    F_r = np.array([[1,2,1],[2,4,2],[1,2,1]])/4
    F_b = F_r
    F_g = np.array([[0,1,0],[1,4,1],[0,1,0]])/4
    r = signal.convolve2d(mosaick_img[...,0], F_r,  mode='same',boundary='symm')
    g = signal.convolve2d(mosaick_img[...,1], F_g,  mode='same',boundary='symm')
    b = signal.convolve2d(mosaick_img[...,2], F_b,  mode='same',boundary='symm')
    recon_img = np.stack([r,g,b], axis=-1)
    return recon_img

def decompress(x):
    # from 3xHxW to 4xH/42W/2
    size = x.shape
    decompressed_image = np.zeros((int(size[0]),int(size[1]),3),dtype=np.uint8)


    decompressed_image[::2,::2, 0] = x[::2,::2]  # G
    decompressed_image[::2,1::2,1] = x[::2, 1::2]  # R
    decompressed_image[1::2,::2,1] = x[1::2, ::2]  # G
    decompressed_image[1::2,1::2,2] = x[1::2, 1::2]  # B
    return decompressed_image
