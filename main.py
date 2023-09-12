import sys

from PyQt5.QtWidgets import QApplication

from model.VGG import *
from VideoStream.camera import *
from predicting import *
from PyQt5 import QtCore, QtGui, QtWidgets
from UI import *

if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

#########################################################################################
    # model = load_model(VGG('VGG19'), './model/VGG19_3.t7')

    # model = VGG('VGG19')
    # net = torch.load('./model/VGG19_3.t7', map_location=device)
    # checkpoint = net['model']
    # model.load_state_dict(checkpoint)
    # model.eval()

    # model = VGG('VGG16')
    # net = torch.load('./model/VGG16_81.t7', map_location=device)
    # checkpoint = net['model']
    # model.load_state_dict(checkpoint)


##############################################################
    # path = "./exp_data/data5.jpg"
    # img, height, width = get_image(path)
    # faces = detect_face(img)
    # cropped_images, _ = detect_landmarks(img, faces, height, width)
    # processed_images = process_img(cropped_images)
    #
    # preds, scores = predicting(processed_images, model)
    # print(preds)
    # print(scores)
######################################################################
    # camera(model)
#####################################################################
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = UI(window)
    window.show()
    sys.exit(app.exec_())


