import argparse
import os

import scipy.misc
from torchvision import transforms
import utilities
from MMNet import *
from ResDNet import *
from data_loaders import *
from problems import *

np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Burst Denoising')

parser.add_argument('-epochs', action="store", type=int, required=True)
parser.add_argument('-depth', action="store", type=int, default=5)
parser.add_argument('-init', action="store_true", dest="init", default=False)
parser.add_argument('-save_images', action="store_true", dest="save_images", default=True)
parser.add_argument('-save_path', action="store", dest="save_path", default='results/')
parser.add_argument('-gpu', action="store_true", dest="use_gpu", default=False)
parser.add_argument('-num_gpus', action="store", dest="num_gpus", type=int, default=1)
parser.add_argument('-max_iter', action="store", dest="max_iter", type=int, default=10)
parser.add_argument('-batch_size', action="store", type=int, required=True)
parser.add_argument('-lr', action="store", dest="lr", type=float, default=0.01)
parser.add_argument('-k1', action="store", dest="k1", type=int, default=5)
parser.add_argument('-k2', action="store", dest="k2", type=int, default=5)
parser.add_argument('-clip', action="store", dest="clip", type=float, default=0.25)
parser.add_argument('-estimate_noise', action="store_true", dest="noise_estimation", default=False)
args = parser.parse_args()
print(args)

epoch_val_inteval = 10
assert args.num_gpus == 1, 'MultiGPU is not supported'
# compile and load pre-trained model
model = ResDNet(BasicBlock, args.depth, weightnorm=True)
model_params = torch.load('pretrained_denoiser/ResDNet_pretrained.pth')
model.load_state_dict(model_params)

if args.noise_estimation:  # in case of noise estimation, continuation scheme is initialized differently
    mmnet = MMNet(model, max_iter=args.max_iter, sigma_max=2, sigma_min=1)
else:
    mmnet = MMNet(model, max_iter=args.max_iter, sigma_max=15, sigma_min=1)

batch_size = args.batch_size

# Create Dataset Loaders for Train and Validation
dataset = lmdbDataset('/home/filippos/data/WD_train_std5_25_v2.lmdb', transform=transforms.Compose([ToTensor()]))

dataloader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

dataset_val = lmdbDataset('/home/filippos/data/WD_val_std1_25_v2.lmdb', transform=transforms.Compose([ToTensor()]))

dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

# create folder for model results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
elif os.path.exists(args.save_path + 'model_best.pth'):
    print('Found model, continue training')
    model_params = torch.load(args.save_path + 'model_best.pth')
    mmnet.load_state_dict(model_params[0])

# store hyper-parameters
with open(args.save_path + 'args.txt', 'wb') as fout:
    fout.write(str.encode(str(args)))
if args.use_gpu:
    if args.num_gpus > 1:
        mmnet = torch.nn.DataParallel(mmnet, device_ids=list(range(args.num_gpus)))
    mmnet = mmnet.cuda()

optimizer = torch.optim.Adam(mmnet.parameters(), lr=args.lr, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
criterion = nn.L1Loss()
runner = TBPTT(mmnet, criterion, args.k1, args.k2, optimizer, max_iter=args.max_iter, clip_grad=None).cuda()

try:
    best_psnr = - np.inf
    for epoch in range(args.epochs):
        mask = None
        # Train model
        psnr_list = []
        mmnet.train()
        for i, sample in enumerate(dataloader_train):

            groundtruth = sample['image_gt'].float()
            image_input = sample['image_input'].float()
            name = sample['filename']
            warp_matrix = sample['warp_matrix'].float()

            p = Burst_Denoise(image_input, warp_matrix)
            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                p.cuda_()

            xcur = runner.train(p, groundtruth, init=args.init, noise_estimation=args.noise_estimation)

            loss = criterion(xcur, groundtruth, p)
            psnr_list += utilities.calc_psnr(xcur / 255, groundtruth / 255)

            del loss, groundtruth, p, xcur, warp_matrix
        torch.cuda.empty_cache()  # release empty memory

        mean_psnr = np.array(psnr_list)
        mean_psnr = mean_psnr.mean()
        print('Epoch[%d/%d] - Train: %.3f' % (epoch, args.epochs, mean_psnr))

        if epoch % epoch_val_inteval != 0:  # evaluate model every pre-defined epochs
            continue

        # evaluate model
        psnr_list = []
        mmnet.eval()
        with torch.no_grad():
            for i, sample in enumerate(dataloader_val):
                groundtruth = sample['image_gt'].float()
                image_input = sample['image_input'].float()
                name = sample['filename']
                warp_matrix = sample['warp_matrix'].float()
                p = Burst_Denoise(image_input, warp_matrix)
                if args.use_gpu:
                    groundtruth = groundtruth.cuda()
                    p.cuda_()

                if args.num_gpus > 1:
                    xcur = mmnet.module.forward_all_iter(p, init=args.init, noise_estimation=args.noise_estimation)
                else:
                    xcur = mmnet.forward_all_iter(p, init=args.init, noise_estimation=args.noise_estimation)

                psnr = utilities.calc_psnr(xcur / 255, groundtruth / 255)
                psnr_list += psnr
                path = args.save_path + 'val/'
                if not os.path.exists(path):
                    os.makedirs(path)

                if args.save_images:
                    xcur = utils.tensor2Im(xcur.cpu())
                    groundtruth = utils.tensor2Im(groundtruth.cpu())
                    for i_ in range(xcur.shape[0]):
                        name_ = name[i_].replace('/', '_')
                        scipy.misc.imsave(path + name_ + '_output.png', xcur[i_].clip(0, 255).astype(np.uint8))
                        scipy.misc.imsave(path + name_ + '_original.png', groundtruth[i_].clip(0, 255).astype(np.uint8))
                        # save burst
                        # NOTE resulting images are big
                        # plot = utilities.plot_batch_burst(torch.stack([p.y[i_],p.get_warped_burst(p.y[i_][None], p.grid[i_][None])[0]]))
                        # plot.savefig(path+name_+'_bursts.png')
                        # plot.clf()
            del groundtruth, image_input, xcur, warp_matrix, p

        mean_psnr = np.array(psnr_list).mean()
        print('Validation:%.3f' % mean_psnr)

        # Save best model
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if mean_psnr > best_psnr:
            print('New best model, saved.')
            if args.num_gpus > 1:
                torch.save([mmnet.module.state_dict(), args.max_iter, args.depth], args.save_path + 'model_best.pth')
            else:
                torch.save([mmnet.state_dict(), args.max_iter, args.depth], args.save_path + 'model_best.pth')
            best_psnr = mean_psnr

        mmnet.train()
        scheduler.step()
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("Detected Keyboard Interrupt, reporting best perfomance on test set.")
# test model
del model, mmnet, runner

torch.cuda.empty_cache()

# load best model configuration
model_params = torch.load(args.save_path + 'model_best.pth')
assert model_params[2] == args.depth

model = ResDNet(BasicBlock, model_params[2], weightnorm=True)
mmnet = MMNet(model, max_iter=model_params[1])
mmnet = mmnet.cuda()
for param in mmnet.parameters():
    param.requires_grad = False

mmnet.load_state_dict(model_params[0])
mmnet = mmnet.cuda()

dataset_test = lmdbDataset('/home/filippos/data/WD_test_std1_25_v2.lmdb', transform=transforms.Compose([ToTensor()]))

datasets = [(dataset_test, 'WD_multiple_stds')]

with torch.no_grad():
    for demosaic_dataset_test, dataset_name in datasets:
        psnr_list = []
        mmnet.eval()
        test_batch_size = 64
        dataloader_test = DataLoader(demosaic_dataset_test, batch_size=test_batch_size,
                                     shuffle=False, num_workers=1, pin_memory=True)
        for i, sample in enumerate(dataloader_test):
            groundtruth = sample['image_gt'].float()
            image_input = sample['image_input'].float()
            name = sample['filename']
            warp_matrix = sample['warp_matrix'].float()

            p = Burst_Denoise(image_input, warp_matrix)
            if args.use_gpu:
                groundtruth = groundtruth.cuda()
                p.cuda_()

            xcur = mmnet.forward_all_iter(p, max_iter=args.max_iter, init=args.init,
                                          noise_estimation=args.noise_estimation)

            psnr = utilities.calc_psnr(xcur / 255, groundtruth / 255)
            psnr_list += psnr
            path = args.save_path + 'test/' + dataset_name + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            if args.save_images:
                xcur = utilities.tensor2Im(xcur.cpu())
                groundtruth = utilities.tensor2Im(groundtruth.cpu())
                for i_ in range(xcur.shape[0]):
                    name_ = name[i_].replace('/', '_')
                    scipy.misc.imsave(path + name_ + '_output.png', xcur[i_].clip(0, 255).astype(np.uint8))
                    scipy.misc.imsave(path + name_ + '_original.png', groundtruth[i_].clip(0, 255).astype(np.uint8))

        mean_psnr = np.array(psnr_list).mean()
        with open(args.save_path + 'results_' + dataset_name + '.txt', 'wb') as fout:
            fout.write(str.encode(str(mean_psnr)))
        print('Test on %s : %.3f' % (dataset_name, mean_psnr))
