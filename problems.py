from modules.wmad_estimator import *


class Problem(nn.Module):
    r""" An abstract Problem Class """

    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.L = torch.FloatTensor(1).fill_(1)

    def task(self):
        return self.task_name

    def energy_grad(self, x):
        pass

    def initialize(self):
        pass

    def cuda_(self):
        pass


class Demosaic(Problem):

    def __init__(self, y, M, estimate_noise=False, task_name='demosaick'):
        r""" Demosaic Problem class
        y is the observed signal
        M is the masking matrix
        """
        Problem.__init__(self, task_name)
        self.y = y
        self.M = M
        if estimate_noise:
            self.estimate_noise()

    def energy_grad(self, x):
        r""" Returns the gradient 1/2||y-Mx||^2
        X is given as input
        """
        return self.M * x - self.y

    def initialize(self):
        r""" Initialize with bilinear interpolation"""
        F_r = torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
        F_b = F_r
        F_g = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
        bilinear_filter = torch.stack([F_r, F_g, F_b])[:, None]
        if self.y.is_cuda:
            bilinear_filter = bilinear_filter.cuda()
        res = F.conv2d(self.y, bilinear_filter, padding=1, groups=3)
        return res

    def estimate_noise(self):
        y = self.y
        if self.y.max() > 1:
            y = self.y / 255
        y = y.sum(dim=1).detach()
        L = Wmad_estimator()(y[:, None])
        self.L = L
        if self.y.max() > 1:
            self.L *= 255  # scale back to uint8 representation

    def cuda_(self):
        self.y = self.y.cuda()
        self.M = self.M.cuda()
        self.L = self.L.cuda()


class Burst_Denoise(Problem):

    def __init__(self, y, A, estimate_noise=False, task_name='burst_denoise'):
        r""" Burst Denoise Problem class
        y is the observed sequence of frames with shape [b, B, C, H, W]
        A is the alignment/warping matrix of each frame to reference. The respective shape is [b, B, 2, 3] and it is
        assumed the affine transformation is the same for each channel.

        Abbreviations for Shape:
        b: batch size
        B: Burst size - Number of Frames in a burst
        C: number of Channels of frames
        H: Height of frames
        W : Width of frames
        """
        Problem.__init__(self, task_name)
        self.y = y
        self.A = A
        self.grid = self.generate_grid(A)  # generate the sampling grid based on the warping matrices
        self.mask = self.generate_pixel_mask(self.grid)
        self.A_inv = self.inverse_affine(self.A)  # calculate the inverse of the warping/affine matrices
        self.grid_inv = self.generate_grid(self.A_inv)  # generate the inverse sampling grid
        if estimate_noise:
            self.estimate_noise()
        self.grad_var = None

    @staticmethod
    def generate_pixel_mask(grid):
        r""" Calculate a binary mask that indicates boundaries
             During warping certain portions of the warped image will be padded to maintain the original shape.
             These padded areas should be taken under consideration during the calculations of energy term gradient,
             because artificial boundaries ('zero', 'reflect') impede with the reconstruction process.
            Return:
                 mask of shape [b, B, 1, H, W]
        """
        # Calculate boundaries across H axis
        # Zero value indicates pixel of synthetic boundary area
        mask_low_x = 1 - (grid[..., 0] < -1).short()
        mask_high_x = 1 - (grid[..., 0] > 1).short()
        # Calculate boundaries across W axis
        mask_low_y = 1 - (grid[..., 1] < -1).short()
        mask_high_y = 1 - (grid[..., 1] > 1).short()

        # Generate masks for the whole image by a
        mask_x = mask_low_x * mask_high_x
        mask_y = mask_low_y * mask_high_y
        mask = mask_x * mask_y
        mask = mask[:, :, None].float()  # expand dimension for broadcasting
        return mask

    @staticmethod
    def inverse_affine(A):
        r""" Calculate the inverse of an affine / warping matrix. """
        A_inv = []
        for i in range(A.shape[1]):
            r_b = A[:, i, :, :2]
            batch_inv = []
            for j, r in enumerate(r_b):
                r_inv = r.inverse()
                t_inv = -1 * r_inv.mv(A[j, i, :, 2])
                batch_inv.append(torch.cat([r_inv, t_inv[:, None]], dim=1))
            batch_inv = torch.stack(batch_inv)
            A_inv.append(batch_inv[:, None])
        return torch.cat(A_inv, dim=1)

    def generate_grid(self, warp_matrix):
        r""" Generate grid using warping matrices.
             Note: OpenCV warping matrices are defined using an integer intexed grid, while Pytorch uses a grid defined
             by [-1, 1]. This function performs the necessary fix.

             Input:
             warp_matrix with shape [b, B, 2, 3]
             Return:
                 grid of shape [b, B, H, W]
         """

        batch, B, C, H, W = self.y.shape

        # create an integer grid / mesh [0, H]x[0, W]
        H_lin = torch.arange(H).float()
        W_lin = torch.arange(W).float()
        H_ = H_lin.expand(H, -1).t()
        W_ = W_lin.expand(W, -1)
        Z = torch.ones(H_.shape).float()  # warping matrices are
        grid = torch.stack([W_, H_, Z])
        grid = torch.stack([grid] * batch)
        aug_warp_matrix = torch.cat([warp_matrix, torch.Tensor([[[[0, 0, 1]]] * B] * batch)], dim=2)
        affine_grids = []

        for i in range(B):
            # apply the affine transformation of each frame on the grid
            affine_grid = torch.bmm(aug_warp_matrix[:, i], grid.reshape(batch, 3, -1)).reshape(grid.shape)
            # correct for difference of representation between OpenCV and Pytorch
            affine_grid[:, 0] = 2 * affine_grid[:, 0] / H - 1
            affine_grid[:, 1] = 2 * affine_grid[:, 1] / W - 1
            affine_grid = affine_grid[:, :2]
            affine_grids.append(affine_grid.permute(0, 2, 3, 1)[:, None])
        return torch.cat(affine_grids, dim=1)

    @staticmethod
    def warp(y, grid):
        r""" Warp frames of y according to grid """
        x = []
        B = y.shape[1] if len(y.shape) > 4 else 0

        if B > 1:
            for i in range(B):
                x.append(F.grid_sample(y[:, i], grid[:, i], padding_mode='zeros')[:, None])
            return torch.cat(x, dim=1)
        else:
            if y.ndimension() == 5 and  y.shape[1] == 1:  # case of number of frames is 1
                return y
            else:
                return F.grid_sample(y, grid[:, 0], padding_mode='zeros')
    def get_warped_burst(self, y, grid):
        return self.warp(y, grid)

    def energy_grad(self, x):
        r""" Returns the gradient of the energy term.

        """
        if self.grad_var is None:
            # calculate adjoint operation on observations
            self.grad_var = self.warp(self.y, self.grid)

        # calculate forward operation on restored image
        x = self.warp(torch.cat([x[:, None]] * self.y.shape[1], dim=1), self.grid_inv)
        # calculate adjoint operation on restored image
        x = self.warp(x, self.grid)
        # Note instead of dividing by the number of observations, we divide by the number of correct cases
        # every boundary case in the mask is represented with zero therefore the summation at that pixel will be less
        # than the number of frames. For valid areas, the summation of ones in the mask will be equal
        # to the number of frames
        grad = (x - self.grad_var).sum(dim=1) / self.mask.sum(dim=1)
        return grad

    def initialize(self):
        r""" Initialize with reference frame."""
        return self.y[:, -1]

    def estimate_noise(self):
        r""" Estimate noise using the reference frame
             Noise is assumed to have the same characteristics across all frames, therefore a single estimation
             is enough.
        """
        y = self.y[:, -1].clone()
        if y.max() > 1:
            y = y / 255
        L = Wmad_estimator()(y)
        self.L = L
        if self.y.max() > 1:
            self.L *= 255  # scale back to uint8 representation
        return L

    def cuda_(self):
        r""" Transfer all needed components to GPU """
        self.y = self.y.cuda()
        self.L = self.L.cuda()
        self.grid = self.grid.cuda()
        self.mask = self.mask.cuda()
        self.grid_inv = self.grid_inv.cuda()


class Burst_Demosaick_Denoise(Problem):

    def __init__(self, y, M, A, estimate_noise=False, task_name='burst_demosaick_denoise'):
        r""" Burst Joint Denoise Demosaick Problem class
        y is the observed sequence of frames with shape [b, B, C, H, W]
        M is the operator that defines the Bayer Pattern with shape [b, B, C, H, W]
        A is the alignment/warping matrix of each frame to reference. The respective shape is [b, B, 2, 3] and it is
        assumed the affine transformation is the same for each channel.

        Abbreviations for Shape:
        b: batch size
        B: Burst size - Number of Frames in a burst
        C: number of Channels of frames
        H: Height of frames
        W : Width of frames
        """
        Problem.__init__(self, task_name)
        self.y = y
        self.A = A  # affine transformation matrix
        self.M = M
        grid = self.generate_grid(A, self.y.shape)  # generate the sampling grid based on the warping matrices
        self.mask = self.generate_pixel_mask(grid)
        self.A_inv = self.inverse_affine(self.A)  # calculate the inverse of the warping/affine matrices
        self.grid_inv = self.generate_grid(self.A_inv, self.y.shape)  # generate the inverse sampling grid

        # In order for the sampling process to be correct, we should account that the image used for sampling is
        # mosaicked. This means that each channel contains either 1/4 or 1/2 of the information and the rest are zeros.
        # This is solved by compressing the information by transforming the image from [3,H,W] to [4,H/2,W/2].
        # The estimated warping matrices should follow the same premise and therefore the translation part of the
        # warping matrix is divided by two, since we have decreased the spatial dimensional in half.
        A_half = self.A.clone()
        A_half[:, :, :, 2] /= 2
        self.grid_warp = self.generate_grid(A_half, [self.y.shape[0], self.y.shape[1], self.y.shape[2] + 1,
                                                     int(self.y.shape[3] / 2), int(self.y.shape[4] / 2)])

        if estimate_noise:
            self.estimate_noise()
        self.grad_var = None

    @staticmethod
    def generate_pixel_mask(grid):
        r""" Calculate a binary mask that indicates boundaries
             During warping certain portions of the warped image will be padded to maintain the original shape.
             These padded areas should be taken under consideration during the calculations of energy term gradient,
             because artificial boundaries ('zero', 'reflect') impede with the reconstruction process.
            Return:
                 mask of shape [b, B, 1, H, W]
        """
        # Calculate boundaries across H axis
        # Zero value indicates pixel of synthetic boundary area
        mask_low_x = 1 - (grid[..., 0] < -1).short()
        mask_high_x = 1 - (grid[..., 0] > 1).short()
        # Calculate boundaries across W axis
        mask_low_y = 1 - (grid[..., 1] < -1).short()
        mask_high_y = 1 - (grid[..., 1] > 1).short()

        # Generate masks for the whole image by a
        mask_x = mask_low_x * mask_high_x
        mask_y = mask_low_y * mask_high_y
        mask = mask_x * mask_y
        mask = mask[:, :, None].float()  # expand dimension for broadcasting
        return mask

    @staticmethod
    def inverse_affine(A):
        r""" Calculate the inverse of an affine / warping matrix. """
        A_inv = []
        for i in range(A.shape[1]):
            r_b = A[:, i, :, :2]
            batch_inv = []
            for j, r in enumerate(r_b):
                r_inv = r.inverse()
                t_inv = -1 * r_inv.mv(A[j, i, :, 2])
                batch_inv.append(torch.cat([r_inv, t_inv[:, None]], dim=1))
            batch_inv = torch.stack(batch_inv)
            A_inv.append(batch_inv[:, None])
        return torch.cat(A_inv, dim=1)

    @staticmethod
    def generate_grid(warp_matrix, shape):
        r""" Generate grid using warping matrices.
             Note: OpenCV warping matrices are defined using an integer intexed grid, while Pytorch uses a grid defined
             by [-1, 1]. This function performs the necessary fix.

             Input:
             warp_matrix with shape [b, B, 2, 3]
             Return:
                 grid of shape [b, B, H, W]
         """

        batch, B, C, H, W = shape
        H_lin = torch.arange(H).float()
        W_lin = torch.arange(W).float()
        H_ = H_lin.expand(H, -1).t()
        W_ = W_lin.expand(W, -1)
        Z = torch.ones(H_.shape).float()
        grid = torch.stack([W_, H_, Z])
        grid = torch.stack([grid] * batch)
        aug_warp_matrix = torch.cat([warp_matrix, torch.Tensor([[[[0, 0, 1]]] * B] * batch)], dim=2)
        affine_grids = []
        for i in range(B):
            affine_grid = torch.bmm(aug_warp_matrix[:, i], grid.reshape(batch, 3, -1)).reshape(grid.shape)
            affine_grid[:, 0] = 2 * affine_grid[:, 0] / H - 1
            affine_grid[:, 1] = 2 * affine_grid[:, 1] / W - 1
            affine_grid = affine_grid[:, :2]
            affine_grids.append(affine_grid.permute(0, 2, 3, 1)[:, None])
        return torch.cat(affine_grids, dim=1)

    def warp(self, y, grid, compress=False):
        r""" Warp frames of y according to grid.
             If y is a mosaicked image then we compress the spatial dimension in order to account for missing
             information.
        """
        x = []
        B = y.shape[1] if len(y.shape) > 4 else 0

        if B > 1:
            for i in range(B):
                if compress:
                    y_ = y[:, i]
                    y_ = self.compress(y_)
                    x.append(F.grid_sample(y_, grid[:, i], padding_mode='zeros')[:, None])
                else:
                    x.append(F.grid_sample(y[:, i], grid[:, i], padding_mode='zeros')[:, None])
            return torch.cat(x, dim=1)
        else:
            if y.ndimension() == 5 and  y.shape[1] == 1:  # case of number of frames is 1
                return y
            else:
                return F.grid_sample(y, grid[:, 0], padding_mode='zeros')

    def get_warped_burst(self, y, grid, compress=True):
        if compress:
            batch, B, C, H, W = y.shape
            res = self.warp(y, grid, True)
            res = res.reshape(batch * B, res.shape[2], res.shape[3], res.shape[4])
            res = self.decompress(res)
            res = res.reshape(batch, B, C, res.shape[2], res.shape[3])
            return res
        else:
            return self.warp(y, grid)

    @staticmethod
    def compress(x):
        r""" Compress a mosaicked image. The sequence according to channels is: R, G1, G2, B

             Input:
                x with shape [b, 3, H, W]
             Return:
                compressed_image of shape [b, 4, H/2, W/2]
         """
        size = x.shape
        if x.is_cuda:
            compressed_image = torch.cuda.FloatTensor(size[0],  # number of batches
                                                      4,  # number of bayer channels
                                                      int(size[2] / 2),  # H
                                                      int(size[3] / 2)).fill_(0)
        else:
            compressed_image = torch.FloatTensor(size[0],  # number of batches
                                                 4,  # number of bayer channels
                                                 int(size[2] / 2),  # H
                                                 int(size[3] / 2)).fill_(0)

        compressed_image[:, 0, :, :] = x[:, 0, ::2, ::2]  # R
        compressed_image[:, 1, :, :] = x[:, 1, ::2, 1::2]  # G
        compressed_image[:, 3, :, :] = x[:, 1, 1::2, ::2]  # G
        compressed_image[:, 2, :, :] = x[:, 2, 1::2, 1::2]  # B
        return compressed_image

    @staticmethod
    def decompress(x):
        r""" Decompress an image.

             Input:
                x with shape [b, 4, H/2, W/2]
             Return:
                decompressed_image of shape [b, 3, H, W]
         """
        size = x.shape
        if x.is_cuda:
            decompressed_image = torch.cuda.FloatTensor(size[0],  # number of batches
                                                        3,  # number of bayer channels
                                                        int(size[2] * 2),  # H
                                                        int(size[3] * 2)).fill_(0)
        else:
            decompressed_image = torch.FloatTensor(size[0],  # number of batches
                                                   3,  # number of bayer channels
                                                   int(size[2]),  # H
                                                   int(size[3])).fill_(0)

        decompressed_image[:, 0, ::2, ::2] = x[:, 0]  # G
        decompressed_image[:, 1, ::2, 1::2] = x[:, 1]  # R
        decompressed_image[:, 1, 1::2, ::2] = x[:, 3]  # G
        decompressed_image[:, 2, 1::2, 1::2] = x[:, 2]  # B
        return decompressed_image

    def energy_grad(self, x):
        r""" Returns the gradient of the energy term. """

        batch, B, C, H, W = self.y.shape
        if self.grad_var is None:
            # Observation y is mosaicked, therefore the adjoint operator is applied on the compressed image.
            # Afterwards the image is restored to original size.
            self.grad_var = self.warp(self.y, self.grid_warp, True)
            self.grad_var = self.grad_var.reshape(batch * B, self.grad_var.shape[2], self.grad_var.shape[3],
                                                  self.grad_var.shape[4])
            self.grad_var = self.decompress(self.grad_var)
            self.grad_var = self.grad_var.reshape(batch, B, C, self.grad_var.shape[2], self.grad_var.shape[3])
        # calculate forward operation on the restored image
        x = self.M * self.warp(torch.cat([x[:, None]] * self.y.shape[1], dim=1), self.grid_inv)
        # calculate adjoint operation on the mosaicked restored image
        x = self.warp(x, self.grid_warp, True)
        x = x.reshape(batch * B, x.shape[2], x.shape[3], x.shape[4])
        x = self.decompress(x)
        x = x.reshape(batch, B, C, x.shape[2], x.shape[3])
        # Note instead of dividing by the number of observations, we divide by the number of correct cases
        # every boundary case in the mask is represented with zero therefore the summation at that pixel will be less
        # than the number of frames. For valid areas, the summation of ones in the mask will be equal
        # to the number of frames
        return (self.mask * (x - self.grad_var)).sum(dim=1) / self.mask.sum(dim=1)

    def initialize(self):
        r""" Initialize with bilinear interpolation of the reference frame"""

        y = self.y[:, -1]  # reference frame is assumed to be always the last
        F_r = torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
        F_b = F_r
        F_g = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
        bilinear_filter = torch.stack([F_r, F_g, F_b])[:, None]  # formulate bilinear filter
        if y.is_cuda:
            bilinear_filter = bilinear_filter.cuda()
        res = F.conv2d(y, bilinear_filter, padding=1, groups=3)

        return res

    def estimate_noise(self):
        r""" Estimate noise using the reference frame
             Noise is assumed to have the same characteristics across all frames, therefore a single estimation
             is enough.
        """
        y = self.y[:, -1]
        if y.max() > 1:
            y = y / 255
        y = y.sum(dim=1).detach()
        L = Wmad_estimator()(y[:, None])
        self.L = L
        if self.y.max() > 1:
            self.L *= 255  # scale back to uint8 representation

    def cuda_(self):
        r""" Transfer all needed components to GPU """
        self.y = self.y.cuda()
        self.L = self.L.cuda()
        self.mask = self.mask.cuda()
        self.grid_inv = self.grid_inv.cuda()
        self.grid_warp = self.grid_warp.cuda()
        self.M = self.M.cuda()
