import argparse
import os
import time
import re
import glob

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

#import transforms
import utils
from conf import settings
from dataset import Promise12
#from metrics import Metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
    parser.add_argument('-wd', type=float, default=0, help='training epoches')
    parser.add_argument('-resume', type=bool, default=False, help='if resume training')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-gpu_ids', nargs='+', type=int, default=0, help='gpu device')
    args = parser.parse_args()
    print(args)



    #root_path = os.path.dirname(os.path.abspath(__file__))

    #checkpoint_path = os.path.join(
    #    root_path, settings.CHECKPOINT_FOLDER, settings.TIME_NOW)
    #log_dir = os.path.join(root_path, settings.LOG_FOLDER, settings.TIME_NOW)

    #if not os.path.exists(checkpoint_path):
    #    os.makedirs(checkpoint_path)
    #checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)

    #writer = SummaryWriter(log_dir=log_dir)

    training_transforms = utils.training_trans()
    test_transforms = utils.test_trans()

    print('loading dataset..........')
    datasets = []
    for list_path in glob.iglob(os.path.join(settings.TRAINING_LIST, '*.txt')):
        fold_idx = os.path.basename(list_path).split('.')[0]
        promise12 = Promise12(list_path, fold_idx)
        datasets.append(promise12)

        #if not istraind(fold_idx):
        #    folds.append(fold_idx)

    print('done')


    for fold in range(len(datasets)):

        fold = str(fold)
        #if istrained(fold):
        #    continue


        # root path
        root_path = os.path.dirname(os.path.abspath(__file__))

        # create checkpoints:
        checkpoint_dir = os.path.join(
            root_path, settings.CHECKPOINT_FOLDER, fold, settings.TIME_NOW)

        # create tensorboard log dirs
        log_dir = os.path.join(root_path, settings.LOG_FOLDER, fold, settings.TIME_NOW)
        writer = SummaryWriter(log_dir=log_dir)

        print(checkpoint_dir)
        print(log_dir)

        # create traing dataset and test datset
        train = []
        test = []
        for dataset in datasets:
            if str(dataset.fold_idx) == fold_idx:
                dataset.transforms = test_transforms
                test.append(dataset)
            else:
                dataset.transforms = training_transforms
                train.append(dataset)

        train_dataset = torch.utils.data.ConcatDataset(train)
        test_dataset = torch.utils.data.ConcatDataset(test)

        train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=args.b, num_workers=4)

        net = utils.get_model(args.net)
        for epoch in range(settings.EPOCH):
            pass




        #count_ones = 0
        #count_zeros = 0

        #for i in range(len(train_dataset)):
        #    _, mask = train_dataset[i]

        #    count_zeros += 1
        #    if mask[mask == 1].sum() > 0:
        #         count_ones += 1
            #zeros = mask[mask == 0].sum()

            #count_ones += ones
            #count_zeros += zeros



        #for i in range(len(test_dataset)):
        #    _, mask = train_dataset[i]

        #    ones = mask[mask == 1].sum()
        #    zeros = mask[mask == 0].sum()

        #    count_ones += ones
        #    count_zeros += zeros


        # print(count_ones / count_zeros)





        #break




    #train_dataset = CamVid(
    #    settings.DATA_PATH,
    #    'train'
    #)
    #valid_dataset = CamVid(
    #    settings.DATA_PATH,
    #    'val'
    #)

    #train_transforms = transforms.Compose([
    #    transforms.RandomRotation(value=train_dataset.ignore_index),
    #    transforms.RandomScale(value=train_dataset.ignore_index),
    #    transforms.RandomGaussianBlur(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ColorJitter(),
    #    transforms.Resize(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD),
    #])

    #valid_transforms = transforms.Compose([
    #    transforms.Resize(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD),
    #])

    #train_dataset.transforms = train_transforms
    #valid_dataset.transforms = valid_transforms

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.b, num_workers=4)
    #validation_loader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=args.b, num_workers=4)

    #net = utils.get_model(args.net, 3, train_dataset.class_num)

    #if args.resume:
    #    weight_path = utils.get_weight_path(
    #        os.path.join(root_path, settings.CHECKPOINT_FOLDER))
    #    print('Loading weight file: {}...'.format(weight_path))
    #    net.load_state_dict(torch.load(weight_path))
    #    print('Done loading!')

    #net = net.cuda()

    #tensor = torch.Tensor(1, 3, *settings.IMAGE_SIZE)
    #utils.visualize_network(writer, net, tensor)

    #optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    #iter_per_epoch = len(train_dataset) / args.b

    #train_scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.e)
    #loss_fn = nn.CrossEntropyLoss()

    #metrics = Metrics(valid_dataset.class_num, valid_dataset.ignore_index)
    #best_iou = 0

    #trained_epochs = 0

    #if args.resume:
    #    trained_epochs = int(
    #        re.search('([0-9]+)-(best|regular).pth', weight_path).group(1))
    #    train_scheduler.step(trained_epochs * len(train_loader))

    #for epoch in range(trained_epochs + 1, args.e + 1):
    #    start = time.time()

    #    net.train()

    #    ious = 0
    #    for batch_idx, (images, masks) in enumerate(train_loader):

    #        optimizer.zero_grad()

    #        images = images.cuda()
    #        masks = masks.cuda()
    #        preds = net(images)

    #        loss = loss_fn(preds, masks)
    #        loss.backward()

    #        optimizer.step()
    #        train_scheduler.step()

    #        print(('Training Epoch:{epoch} [{trained_samples}/{total_samples}] '
    #                'Lr:{lr:0.6f} Loss:{loss:0.4f} Beta1:{beta:0.4f}').format(
    #            loss=loss.item(),
    #            epoch=epoch,
    #            trained_samples=batch_idx * args.b + len(images),
    #            total_samples=len(train_dataset),
    #            lr=optimizer.param_groups[0]['lr'],
    #            beta=optimizer.param_groups[0]['betas'][0]
    #        ))

    #        n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
    #        utils.visulaize_lastlayer(
    #            writer,
    #            net,
    #            n_iter,
    #        )

    #    utils.visualize_scalar(
    #        writer,
    #        'Train/LearningRate',
    #        optimizer.param_groups[0]['lr'],
    #        epoch,
    #    )

    #    utils.visualize_scalar(
    #        writer,
    #        'Train/Beta1',
    #        optimizer.param_groups[0]['betas'][0],
    #        epoch,
    #    )
    #    utils.visualize_param_hist(writer, net, epoch)
    #    print('time for training epoch {} : {}'.format(epoch, time.time() - start))

    #    net.eval()
    #    test_loss = 0.0

    #    with torch.no_grad():
    #        for batch_idx, (images, masks) in enumerate(validation_loader):

    #            images = images.cuda()
    #            masks = masks.cuda()

    #            preds = net(images)

    #            loss = loss_fn(preds, masks)
    #            test_loss += loss.item()

    #            preds = preds.argmax(dim=1)
    #            preds = preds.view(-1).cpu().data.numpy()
    #            masks = masks.view(-1).cpu().data.numpy()
    #            metrics.add(preds, masks)
    #            n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1

    #    miou = metrics.iou()
    #    precision = metrics.precision()
    #    recall = metrics.recall()
    #    metrics.clear()

    #    utils.visualize_scalar(
    #        writer,
    #        'Test/mIOU',
    #        miou,
    #        epoch,
    #    )

    #    utils.visualize_scalar(
    #        writer,
    #        'Test/Loss',
    #        test_loss / len(valid_dataset),
    #        epoch,
    #    )

    #    eval_msg = (
    #        'Test set Average loss: {loss:.4f}, '
    #        'mIOU: {miou:.4f}, '
    #        'recall: {recall:.4f}, '
    #        'precision: {precision:.4f}'
    #    )

    #    print(eval_msg.format(
    #        loss=test_loss / len(valid_dataset),
    #        miou=miou,
    #        recall=recall,
    #        precision=precision
    #    ))

    #    if best_iou < miou and epoch > args.e // 2:
    #        best_iou = miou
    #        torch.save(net.state_dict(),
    #                        checkpoint_path.format(epoch=epoch, type='best'))
    #        continue

    #    if not epoch % settings.SAVE_EPOCH:
    #        torch.save(net.state_dict(),
    #                        checkpoint_path.format(epoch=epoch, type='regular'))