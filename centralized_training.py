import os
import torch
import numpy as np
from data_utils.data_module import read_label2, read_image, CustomImageDataset
from loss_utils.cross_entropy_loss import CrossEntropy2d
from neural_nets import SegUNet
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    print("Loading AIS data..")

    train_path = os.path.join('..', 'Data', 'AIS_Data', 'Central')
    train_img = read_image(os.path.join(train_path, 'image'))
    train_labels = read_label2(os.path.join(train_path, 'label'))

    validation_path = os.path.join('..', 'Data', 'AIS_Data', 'Val')
    validation_img = read_image(os.path.join(validation_path, 'image'))
    validation_labels = read_label2(os.path.join(validation_path, 'label'))

    print("AIS data loading finished")

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(train_img, train_labels), batch_size=4,
                                               shuffle=True, num_workers=20)
    val_loader = torch.utils.data.DataLoader(CustomImageDataset(validation_img, validation_labels), batch_size=1,
                                             shuffle=False, num_workers=20)
    device = 'cuda'

    model = SegUNet().to(device)
    # checkpoint_path = os.path.join('result', 'central', 'model_checkpoint_480.pth.tar')
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint)

    loss_func = CrossEntropy2d()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    writer = SummaryWriter(comment=' Central with GN ResUNet')

    print("Start training..")
    epoch_loss = []
    for iter in range(500):
        model.train()
        batch_loss = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).long()
            model.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print("Loss at epoch {} is {:.3f}".format(iter, avg_loss))
        writer.add_scalar('Loss/Train', avg_loss, iter)
        scheduler.step()

        if iter % 30 == 0:
            print("Evaluating...")
            model.eval()
            batch_loss = []
            val_scores = []
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device).long()
                pred = model(x)

                _, val_preds = torch.max(pred, 1)
                val_scores.append(np.mean((val_preds == y).data.cpu().numpy()))

                with torch.no_grad():
                    loss = loss_func(pred, y)

                batch_loss.append(loss.item())
            avg_loss = sum(batch_loss) / len(batch_loss)
            avg_acc = np.mean(val_scores)
            print("Validation Loss at epoch {} is {:.3f}".format(iter, avg_loss))
            print("Validation pixel accuracy at epoch {} is {:.3f}".format(iter, avg_acc))

            writer.add_scalar('Loss/Validation', avg_loss, iter)
            writer.add_scalar('Accuracy/Validation', np.mean(val_scores), iter)

            checkpoint_dir = os.path.join("result", "central")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_name = os.path.join(checkpoint_dir, 'model_checkpoint_{}.pth.tar'.format(iter))
            torch.save(model.state_dict(), checkpoint_name)
            print('\r[INFO] Checkpoint has been saved: %s\n' % checkpoint_name)

    writer.close()
