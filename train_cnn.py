import os.path
import shutil

import torch.optim
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import ToTensor
from dataset import CropDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from example import get_args
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from visual import plot_confusion_matrix






if __name__ == '__main__':

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_epochs = args.epochs
    batch_size = args.batch_size
    image_size = args.images_size
    root = args.root


    train_transform = Compose([

        Resize((image_size, image_size)),
        ToTensor(),

    ])
    test_transform = Compose([

        Resize((image_size, image_size)),
        ToTensor(),

    ])


    train_dataset = CropDataset(root=root, train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = CropDataset(root=root, train=False,transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)


    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)


    writter = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=2).to(device)






    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)

    if args.checkpoints:
        checkpoint = torch.load(args.checkpoints)
        start_epoch = checkpoint["epoch"]

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
    best_acc = 0




    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader,colour='cyan')
        for iter, (images, labels) in enumerate(progress_bar):

            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            writter.add_scalar("Train/Loss", loss_value,epoch* num_iters + iter)

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]

        plot_confusion_matrix(writter,confusion_matrix(all_labels,all_predictions), class_names=test_dataset.categories,epoch=epoch)



        accuracy = accuracy_score(all_labels,all_predictions)

        print("Epoch {}: Accuracy: {:.2f}".format(epoch+1,accuracy))
        writter.add_scalar("Validation/Accuracy", accuracy,epoch)
        #torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))

        checkpoint = {
            "epoch": epoch + 1,

            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))

        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,

                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
        #print(classification_report(all_labels, all_predictions))
