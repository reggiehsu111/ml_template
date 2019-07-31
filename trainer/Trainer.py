from torch.utils.data import DataLoader
from terminaltables import AsciiTable
import torch
import time

class Trainer():
    # initialize trainer with args
    def __init__(self, dataloaders, model, visualizer, opt):
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.train_dataset_size = len(self.train_dataloader)
        self.model = model
        self.visualizer = visualizer
        self.opt = opt
        if self.opt.args.model == 'yolo':
            self.metrics = [
                "grid_size",
                "loss",
                "x",
                "y",
                "w",
                "h",
                "conf",
                "cls",
                "cls_acc",
                "recall50",
                "recall75",
                "precision",
                "conf_obj",
                "conf_noobj",
            ]



        # set if azure log is required
        if self.opt.args.azure_log:
            # log to Azure
            from azureml.core.run import Run
            self.azure_run = Run.get_context()
            self.azure_run.log('batch_size', np.float(self.opt.args.batch_size))
            self.azure_run.log('lr', np.float(self.opt.args.lr))

    # training logic
    def train(self):
        total_iters = 0                # the total number of training iterations
        print("Size of training dataset: ", len(self.train_dataloader))
        
        for epoch in range(self.opt.args.epoch_count, self.opt.args.niter + self.opt.args.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            # Only use train_dataloader for training
            for i, data in enumerate(self.train_dataloader):  # inner loop within one epoch
                if i == 500:
                    print("break loop")
                    break
                # print time passed every print_freq
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.args.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                self.visualizer.reset()
                total_iters += self.opt.args.batch_size
                epoch_iter += self.opt.args.batch_size


                self.model.set_input(data)         # unpack data from dataloader and apply preprocessing
                self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if self.opt.args.has_visuals and total_iters % self.opt.args.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.args.update_html_freq == 0
                    self.model.compute_visuals()
                    self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                if total_iters % self.opt.args.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.args.batch_size
                    self.visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.args.display_id > 0:
                        self.visualizer.plot_current_losses(epoch, float(epoch_iter) / self.train_dataset_size, losses)

                if total_iters % self.opt.args.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if self.opt.args.save_by_iter else 'latest'
                    self.model.save_networks(save_suffix)

                if self.opt.args.azure_log:
                    self.azure_run.log('best_validate_acc', best_validate_acc)
                    self.azure_run.log('saved_model_testing_acc', saved_model_testing_acc)
                    self.azure_run.log('train_acc', training_acc)
                    self.azure_run.log('train_loss', training_loss)
                    self.azure_run.log('valid_acc', validate_acc)
                    self.azure_run.log('valid_loss', validation_loss)
                    self.azure_run.log('test_acc', testing_acc)
                    self.azure_run.log('test_loss', testing_loss)

                if self.opt.args.model == 'yolo':

                    # Yolo specific
                    metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(self.model.netYolo.yolo_layers))]]]
                    # Log metrics at each YOLO layer
                    for i, metric in enumerate(self.metrics):
                        formats = {m: "%.6f" for m in self.metrics}
                        formats["grid_size"] = "%2d"
                        formats["cls_acc"] = "%.2f%%"
                        row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.model.netYolo.yolo_layers]
                        metric_table += [[metric, *row_metrics]]
                    print(AsciiTable(metric_table).table)

                iter_data_time = time.time()
            if epoch % self.opt.args.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.args.niter + self.opt.args.niter_decay, time.time() - epoch_start_time))
            self.model.update_learning_rate()