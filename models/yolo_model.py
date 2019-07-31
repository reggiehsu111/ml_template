import torch
from torch.utils.data import random_split
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .yolo_networks import *
from util.util import weights_init_normal
from .networks import *


class YOLOModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--model_config', type=str, default='options/yolov3.cfg', help='configuration for yolov3 model')

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        # define losses to plot
        self.loss_names = []
        self.layer_losses = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'cls_acc', 'recall50', 'recall75', 'precision', 'conf_obj', 'conf_noobj', 'grid_size']
        for x in range(3):
            for loss in self.layer_losses:
                self.loss_names.append("layer" + str(x) + "_" + loss)

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_name = ['predictions']

        self.model_names = ['Yolo']

        self.netYolo = Darknet(opt.args.model_config).to(self.device)
        self.netYolo.apply(weights_init_normal)

        self.optimizer = torch.optim.Adam(self.netYolo.parameters(), lr=opt.args.lr, betas=(opt.args.beta1, 0.999))
        self.optimizers.append(self.optimizer)
        self.netYolo = init_net(self.netYolo, opt.args.init_type, opt.args.init_gain, opt.args.gpu_ids)

    def set_input(self, input):
        self.img_path = input[0]
        self.img = input[1].to(self.device)
        self.targets = input[2].to(self.device)


    def forward(self):
        self.loss, self.outputs, self.yolo_layer_metrics = self.netYolo(self.img, self.targets)
        i = 0
        for layer in self.yolo_layer_metrics:
            for k in layer.keys():
                loss_str = "loss_layer"+ str(i) + "_" + k
                setattr(self, loss_str, layer[k])
            i += 1

        

    def optimize_parameters(self):
        self.forward()
        self.loss.backward()



