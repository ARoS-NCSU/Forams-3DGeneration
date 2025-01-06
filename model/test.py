import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets


def test(dataloader, model, epoch, opt):

    cudnn.benchmark = True
    alpha = 0

    # model = torch.load(os.path.join(
    #     opt.model_save_path, '{0}/model_epoch_{1}.pth'.format(opt.model_save_path, epoch)
    # ))

    model = model.eval()
    # model = model.to(opt.device)

    n_total = 0
    n_correct = 0

    for _, data_target in enumerate(dataloader):

        # test model using target data
        t_img, t_label = data_target

        batch_size = len(t_label)


        t_img = t_img.to(opt.device)
        t_label = t_label.to(opt.device)

        class_output, _ = model(input_data=t_img, alpha=alpha)
        # pred = class_output.data.max(1, keepdim=True)[1]
        pred = class_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size



    accu = n_correct.data.numpy() * 1.0 / n_total
    print("Accuracy: ",epoch, accu)

    opt.plotter.plot('acc_class', opt.plot_legend, 'Class Accuracy', epoch, accu)

    # print('epoch: %d, accuracy of the val dataset: %f' % (epoch, accu))