import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import tqdm

from model import CNNModel
from dataloader import load_dataset
from opt import opts
from test import test
import utils

torch.manual_seed(0)
np.random.seed(0)


def main(opt):

    train_set_real = load_dataset(opt, opt.train_real, img_format = 'jpg', domain = 'real')
    dataloader_source = data_utils.DataLoader(train_set_real, batch_size=opt.batch_size, shuffle=True,  num_workers=4)

    train_set_synthetic = load_dataset(opt, opt.train_synthetic, img_format = 'png', domain = 'synthetic', train_len = len(train_set_real))
    dataloader_target = data_utils.DataLoader(train_set_synthetic, batch_size=opt.batch_size, shuffle=True,  num_workers=4)

    test_dataset = load_dataset(opt, opt.test_real, img_format = 'jpg', domain = 'real')
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,  num_workers=4)

    print("\nNumber of real training samples in each class:")
    train_set_real.class_distribution()
    print("\nNumber of synthetic training samples:")
    train_set_synthetic.class_distribution()


    model = CNNModel(opt).to(opt.device)
    # loss_class = torch.nn.CrossEntropyLoss().to(opt.device)
    # loss_domain = torch.nn.CrossEntropyLoss().to(opt.device)
    loss_class = torch.nn.NLLLoss().to(opt.device)
    loss_domain = torch.nn.NLLLoss().to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)

    len_dataloader = min(len(dataloader_source), len(dataloader_target))


    losses_class = utils.AverageMeter()
    losses_domain = utils.AverageMeter()

    print("\nTraining with Domain Adaptation Head: ", opt.DOMAIN_ADAPTATION_HEAD)


    for epoch in tqdm.tqdm(range(opt.epochs)):

        if opt.DOMAIN_ADAPTATION_HEAD:
            dataloader = zip(dataloader_source, dataloader_target)
        else:
            dataloader = dataloader_source

        for i, data in enumerate(dataloader):
        
            if opt.DOMAIN_ADAPTATION_HEAD:
                data_source, data_target = data
            else:
                data_source = data
            
            p = float(i + epoch * len_dataloader) / opt.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            s_img, s_label = data_source
            s_img = s_img.to(opt.device)
            s_label = s_label.to(opt.device)
            batch_size = len(s_label)

            model.zero_grad()
            class_output, domain_output = model(input_data=s_img, alpha=alpha, DOMAIN_ADAPTATION_HEAD = opt.DOMAIN_ADAPTATION_HEAD)
            err = F.cross_entropy(class_output, s_label)
            losses_class.update(err.data.cpu().numpy(), s_label.size(0))

            if opt.DOMAIN_ADAPTATION_HEAD:
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()
                domain_label = domain_label.to(opt.device)
        
                err_s_domain = F.cross_entropy(domain_output, domain_label)

                # training model using target data
                t_img, _ = data_target

                batch_size = len(t_img)

                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()
                domain_label = domain_label.to(opt.device)

                t_img = t_img.to(opt.device)

                _, domain_output = model(input_data=t_img, alpha=alpha, DOMAIN_ADAPTATION_HEAD = opt.DOMAIN_ADAPTATION_HEAD)
                err_t_domain = F.cross_entropy(domain_output, domain_label)
                losses_domain.update(err_s_domain.data.cpu().numpy() + err_t_domain.data.cpu().numpy(), domain_label.size(0) + s_label.size(0))

                err = opt.domain_weight * (err_t_domain + err_s_domain) + (1 - opt.domain_weight) * err
                # err += err_t_domain + err_s_domain

            err.backward()
            optimizer.step()

            # print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
            #     % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
            #         err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))

        opt.plotter.plot('loss_class', opt.plot_legend, 'Class Loss', epoch, losses_class.avg)
        if opt.DOMAIN_ADAPTATION_HEAD:
            opt.plotter.plot('loss_domain', opt.plot_legend, 'Domain Loss', epoch, losses_domain.avg)

        # torch.save(model, '{0}/model_epoch_{1}.pth'.format(opt.model_save_path, epoch))
        if epoch % opt.val_intervals == 0:
            test(test_dataloader, model, epoch, opt)


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)