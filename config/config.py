import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=3473)

    # stage hyper parameters:
    parser.add_argument('--nfc', type=int, help='number of filters per conv layer', default=128)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers per stage', default=3)
    parser.add_argument('--padd_size', type=int, help='padd size', default=0)

    # pyramid parameters:
    parser.add_argument('--nc_im', type=int, help='number of channels of img', default=3)
    parser.add_argument('--min_size', type=int,help='image minimal size at the coarser scale',default=250)
    parser.add_argument('--max_size', type=int,help='image maximum size at the coarser scale', default=500)
    parser.add_argument('--scale_factor', type=float, help='scale factor of training', default=0.75)
    parser.add_argument('--scale_num', type=int, help='stage number of training', default=1)

    # training parameters:
    parser.add_argument('--losses', type=str, default=['ssim11', 'mse'], help='losses')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of dataset')
    parser.add_argument('--num_data', type=int, default=100, help='total size of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--niter', type=int, default=500, help='number of iteration per scale')
    parser.add_argument('--pixel_shuffle_p', type=float, default=0.005, help='pixel shuffle rate (0.005 => 0.5%)')
    # parser.add_argument('--', type=, help='', default= )

    return parser