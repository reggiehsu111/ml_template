import argparse
import os
from util import util
import models
import data
import torch
from os.path import expanduser
# set home directory to find absolute path
home = expanduser("~")

# Base options that can be used in every scenario
class BaseOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Cyclegan network")
        # name of the experiment
        parser.add_argument('--name', type=str, default='yolo_experiment', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument("--batch_size", type=int, default=2, help="input batch size.")
        parser.add_argument("--num_workers", type=int, default=1, help="number of threads.")
        parser.add_argument('--dataroot', type=str, default='../PyTorch-YOLOv3/data/coco/trainvalno5k.txt', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        # parser.add_argument('--dataroot', type=str, default='datasets/maps', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='yolo', help='chooses which model to use. [yolo | cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--dataset_mode', type=str, default='list', help='chooses how datasets are loaded. [list | unaligned | aligned | single | colorization]')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping  of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.args.name = opt.args.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        # parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        # parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.parser = parser

        


    def print_options(self):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / args.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.args.checkpoints_dir, self.args.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_args.txt'.format(self.args.phase))
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """

        # get the basic options
        args, _ = self.parser.parse_known_args()

        # modify model-related parser options
        model_name = args.model
        print("model name: ", model_name)
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(self.parser, args.isTrain)
        argst, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = args.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, args.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        args = self.gather_options()
        self.args = args
        args.isTrain = self.args.isTrain   # train or test

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.name = args.name + suffix

        

        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])

        self.args = args
        self.print_options()
        return self