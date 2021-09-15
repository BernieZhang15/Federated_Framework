import os
import torch
from data_utils.data_module import read_label2, read_image, CustomImageDataset
from loss_utils.focol_tversky_loss import TverskyCrossEntropyDiceWeightedLoss
from neural_nets import SegUNet
from tensorboardX import SummaryWriter
from metric_utils.metric import Metric

if __name__ == "__main__":

    print("Loading AIS data..")

    train_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Central')
    train_img = read_image(os.path.join(train_path, 'image'))
    train_labels = read_label2(os.path.join(train_path, 'label'))

    cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    validation_data = {}
    validation_label = {}

    for city in cities:
        validation_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Val', city)
        val_img_sets = read_image(os.path.join(validation_path, 'image'))
        validation_data[city] = val_img_sets
        val_label_sets = read_label2(os.path.join(validation_path, 'label'))
        validation_label[city] = val_label_sets

    print("AIS data loading finished")

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(train_img, train_labels), batch_size=4,
                                               shuffle=True, num_workers=20)
    val_loader = {city: torch.utils.data.DataLoader(CustomImageDataset(validation_data[city], validation_label[city]),
                                                    batch_size=2, shuffle=True, num_workers=10) for city in cities}
    device = 'cuda'

    model = SegUNet().to(device)

    loss_func = TverskyCrossEntropyDiceWeightedLoss(num_class=2, device=device)

    metric = Metric(num_class=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    writer = SummaryWriter(comment=' Building_Extraction_Central_TverskyLoss')

    print("Start training..")
    epoch_loss = []
    for iter in range(300):
        model.train()
        batch_loss = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            pred, fm = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print("Loss at epoch {} is {:.3f}".format(iter, avg_loss))
        writer.add_scalar('Loss/Train', avg_loss, iter)
        scheduler.step()

        if iter % 20 == 0:
            print("Evaluating...")
            model.eval()
            for city in cities:
                metric.reset()
                for batch_idx, (x, y) in enumerate(val_loader[city]):
                    x, y = x.to(device), y.to(device).long()
                    pred, fm = model(x)
                    _, pred = torch.max(pred, 1)
                    metric.add_pixel_accuracy(pred, y)
                    metric.add_confusion_matrix(pred, y)

                iou, mean_iou = metric.iou_value()
                accuracy = metric.accuracy_value()
                writer.add_scalar('Accuracy/' + city, accuracy, iter)
                writer.add_scalar('Building IoU/' + city, iou[1], iter)
                writer.add_scalar('Background IoU/' + city, iou[0], iter)

            # checkpoint_dir = os.path.join("result", "central")
            # if not os.path.exists(checkpoint_dir):
            #     os.makedirs(checkpoint_dir)
            # checkpoint_name = os.path.join(checkpoint_dir, 'model_checkpoint_{}.pth.tar'.format(iter))
            # torch.save(model.state_dict(), checkpoint_name)
            # print('\r[INFO] Checkpoint has been saved: %s\n' % checkpoint_name)

    writer.close()
