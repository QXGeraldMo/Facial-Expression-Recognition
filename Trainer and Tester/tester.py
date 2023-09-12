import seaborn as sn
from sklearn.metrics import confusion_matrix
from data.fer2013plus import *
from model.VGG import *


def test_loop(model, test_loader):
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model.eval()
    model = model.to(device)
    #     test_loss = 0
    correct = 0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in test_loader:
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

            #             test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            predicted_labels.append(pred)
            true_labels.append(targets)

            true_labels_tensor = torch.tensor(true_labels)
            predicted_labels_tensor = torch.tensor(predicted_labels)

            correct += pred.eq(targets.view_as(pred)).sum().item()

    #     test_loss /= len(test_loader.dataset)

    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    matrix = confusion_matrix(true_labels_tensor.cpu().numpy(), predicted_labels_tensor.cpu().numpy())
    plot = sn.heatmap(np.array(matrix),
                      annot=True,
                      annot_kws={"size": 16},
                      fmt='g',
                      cbar=False,
                      xticklabels=['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear',
                                   'contempt'],
                      yticklabels=['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear',
                                   'contempt'])
    plot.set(xlabel='Predicted', ylabel='Actual')
    print(plot)


########################################################################################
########################################################################################
model = VGG16(num_classes=8)

net = torch.load('../model/VGG16_81.t7.t7')
checkpoint = net['model']
model.load_state_dict(checkpoint)

test_dataset = Fer2013(csv_file='../data/FER2013plus/FER2013Test/label.csv',
                       root_dir='../data/FER2013plus/FER2013Test/',
                       transform=transforms.Compose([ToTensor()]),
                       mode="test")


test_loader = DataLoader(test_dataset)
test_loop(model=model, test_loader=test_loader)
