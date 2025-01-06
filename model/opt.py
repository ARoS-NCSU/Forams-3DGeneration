import torch
import argparse
import os
import sys
import utils

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    self.parser.add_argument('--lr', type=float, default=0.01)
    self.parser.add_argument('--momentum', type=float, default=0.5)
    self.parser.add_argument('--epochs', type=int, default=200)
    self.parser.add_argument('--val_intervals', type=int, default=2)
    self.parser.add_argument('--batch_size', type=int, default=32)

  def parse(self):
    opt = self.parser.parse_args()

    use_cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")

    print("Training on ", opt.device)

    opt.DOMAIN_ADAPTATION_HEAD = True
    opt.train_val_split = 0.8    
    opt.momentum = 0.0
    opt.test_batch_size = 1024
    opt.train_real = 'C:\\Users\\sport\\Google Drive\\3 Semester\\Individual Study\\synthetic_foram_model\\data\\real_preprocessed2\\'
    opt.train_synthetic = 'C:\\Users\\sport\\Google Drive\\3 Semester\\Individual Study\\synthetic_foram_model\\data\\synthetic\\'
    opt.test_real = 'C:\\Users\\sport\\Google Drive\\3 Semester\\Individual Study\\synthetic_foram_model\\data\\test2\\'
    opt.image_size = 128
    opt.n_classes = 6
    labels = os.listdir(opt.train_real)
    opt.label_dict = {k : i for i, k in enumerate(labels)}
    opt.model_save_path = 'C:\\Users\\sport\\Google Drive\\3 Semester\\Individual Study\\synthetic_foram_model\\data\\model\\'
    opt.domain_weight = 0.2

    # opt.plot_legend = 'w_ DA_CC_0.2'
    opt.plot_legend = 'Experiment'

  
    opt.plotter = utils.VisdomLinePlotter(env_name=opt.plot_legend)
    return opt


if __name__ == "__main__":
    opt = opts().parse()