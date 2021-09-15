import os
import torch
import random
from data_utils.data_module import read_label2, read_image, CustomImageDataset
from loss_utils.focol_tversky_loss import TverskyCrossEntropyDiceWeightedLoss
from neural_nets import SegUNet
from tensorboardX import SummaryWriter
from metric_utils.metric import Metric


if __name__ == "__main__":

    device = 'cuda'
    writer = SummaryWriter(comment=' Building Extraction Sequence Learning')
    loss_func = TverskyCrossEntropyDiceWeightedLoss(num_class=2, device=device)
    cities_train = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    cities_val = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    print("Loading Aerial Dataset..")
    train_loader = {}
    val_loader = {}

    for city in cities_train:
        train_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Train', city)
        train_img = read_image(os.path.join(train_path, 'image'))
        train_label = read_label2(os.path.join(train_path, 'label'))
        train_loader[city] = torch.utils.data.DataLoader(CustomImageDataset(train_img, train_label), batch_size=4,
                                                         shuffle=True, num_workers=20)

        if city in cities_val:
            validation_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Val', city)
            val_img = read_image(os.path.join(validation_path, 'image'))
            val_label = read_label2(os.path.join(validation_path, 'label'))
            val_loader[city] = torch.utils.data.DataLoader(CustomImageDataset(val_img, val_label),
                                                 batch_size=2, shuffle=True, num_workers=20)
    print("Aerial Dataset loading finished")

    print("Start training..")
    model = SegUNet().to(device)
    metric = Metric(num_class=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    for iter in range(300):
        random.shuffle(cities_train)
        model.train()
        running_loss = 0.0
        for city in cities_train:
            for batch_idx, (x, y) in enumerate(train_loader[city]):
                x, y = x.to(device), y.to(device).long()
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        running_loss = running_loss / len(train_loader) / len(cities_train)
        print("Loss at epoch {} is {:.3f}".format(iter, running_loss))
        scheduler.step()

        if iter % 20 == 0:
            print("Evaluating...")
            model.eval()

            for city in cities_val:
                metric.reset()
                with torch.no_grad():
                    for batch_idx, (x, y) in enumerate(val_loader[city]):
                        x, y = x.to(device), y.to(device).long()
                        pred = model(x)
                        _, pred = torch.max(pred, 1)
                        metric.add_pixel_accuracy(pred, y)
                        metric.add_confusion_matrix(pred, y)

                iou, mean_iou = metric.iou_value()
                accuracy = metric.accuracy_value()
                writer.add_scalar('Accuracy/' + city, accuracy, iter)
                writer.add_scalar('Building IoU/' + city, iou[1], iter)
                writer.add_scalar('Background IoU/' + city, iou[0], iter)

    torch.cuda.empty_cache()
    writer.close()
