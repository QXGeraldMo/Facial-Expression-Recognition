import torch.optim as optim
import matplotlib.pyplot as plt
from data.fer2013plus import *
from model.VGG import *
import datetime

epoch_print_gap = 1


def train_loop(config, checkpoint_dir=None, n_epochs=0, model=None, train_data=None, val_data=None):
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'],
                          momentum=config['momentum'])

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=config["batch_size"])
    val_loader = DataLoader(val_data)
    model = model.to(device)

    best_val_acc = 0
    best_val_epoch = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    path = '../model'

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct_train = 0
        train_steps = 0
        total_train = 0
        model.train()

        if epoch >= 50:
            for group in optimizer.param_groups:
                factor = 0.9 ** ((epoch - 80) // 5)
                group['lr'] = group['lr'] * factor

        for data in train_loader:

            imgs = data['Image']
            labels = data['Label']
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            #             correct_train += predicted.eq(labels.data).sum()

            train_steps += 1

        train_acc = 100 * correct_train / total_train
        train_losses.append(loss_train / train_steps)
        train_accs.append(train_acc)

        if epoch == 1 or epoch % epoch_print_gap == 0:
            print('{} Epoch {}, Training loss {}, Training Accuracy {}'.format(
                datetime.datetime.now(), epoch, train_losses[-1], train_acc))

        # Evaluation
        loss_val = 0
        val_steps = 0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs = data['Image']
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                targets = data['Label']

                inputs, targets = inputs.to(device), targets.to(device)
                #                 inputs, targets = inputs.to(device), targets.type(torch.FloatTensor).to(device)

                outputs = model(inputs)
                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops

                loss = criterion(outputs, targets)
                loss_val += loss.cpu().numpy()

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets.data).sum().item()
                #                 correct_val += predicted.eq(targets.data).sum()

                #                 val_loss += loss.cpu().numpy()
                val_steps += 1

            val_acc = 100 * correct_val / total_val
            val_losses.append(loss_val / val_steps)
            val_accs.append(val_acc)

            validtaion_loss = loss_val / val_steps
            print('Validation loss {}, Validation accuracy {}'.format(val_losses[-1], val_acc))

            if val_acc > best_val_acc:
                print("saving..")
                print("best_PublicTest_acc: %0.3f" % val_acc)

                state = {
                    'model': model.state_dict() if use_cuda else model,
                    'acc': val_acc,
                    'epoch': epoch,
                }

                if not os.path.isdir(path):
                    os.mkdir(path)

                torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
                best_val_acc = val_acc
                best_val_epoch = epoch

    # Plotting loss and accuracy curves
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.legend()
    plt.show()


######################################################################
train_dataset = Fer2013(csv_file='./FER2013plus/FER2013Train/label.csv',
                        root_dir='./FER2013plus/FER2013Train',
                        transform=transforms.Compose([ToTensor()]),
                        mode="train")

validation_dataset = Fer2013(csv_file='./FER2013plus/FER2013Valid/label.csv',
                             root_dir='./FER2013plus/FER2013Valid',
                             transform=transforms.Compose([ToTensor()]),
                             mode = 'val')

model = VGG16(num_classes=8)


n_epochs = 40
config = {'lr': 0.01, 'batch_size': 128, 'weight_decay': 5e-4, 'momentum': 0.9}
train_loop(
    config = config,
    n_epochs = n_epochs,
    model = model,
    train_data = train_dataset,
    val_data = validation_dataset,
)
