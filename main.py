import time
from options.TrainOptions import TrainOptions
from data import create_dataloader
from trainer.Trainer import Trainer
from models import create_model
from util.visualizer import Visualizer

def main():
    opt = TrainOptions().parse()

    # create dataloaders for each phase
    dataloaders = create_dataloader(opt)

    print("type of subset: ", type(dataloaders[0]))

    # Create model
    model = create_model(opt)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # initialize trainer
    trainer = Trainer(dataloaders, model, visualizer, opt)
    trainer.train()



if __name__ == '__main__':
    main()