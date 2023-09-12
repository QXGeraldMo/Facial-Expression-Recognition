from PyQt5.QtGui import QPainter, QPen
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtChart import *
import cv2
from predicting import *
from face_alignment import *
from VideoStream.camera import *
import pyqtgraph as pg
from model.VGG import *
from functools import partial


class UI(object):
    def __init__(self, window):

        pg.setConfigOption('background', '#19232D')
        pg.setConfigOption('foreground', 'd')
        pg.setConfigOptions(antialias=True)

        self.setupUi(window)

        device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")
        model = VGG16(num_classes=8)
        net = torch.load('./model/VGG16_81.t7', map_location=device)
        checkpoint = net['model']
        model.load_state_dict(checkpoint)
        self.model = model

    def setupUi(self, window):
        window.setObjectName("Facial Expression Recognition")
        window.resize(1200, 800)
        palette1 = QtGui.QPalette()
        palette1.setBrush(window.backgroundRole(), QtGui.QBrush(QtGui.QPixmap('./bg.jpg')))
        window.setPalette(palette1)
        window.setAutoFillBackground(True)

        self.centralwidget = QtWidgets.QWidget(window)
        self.centralwidget.setObjectName("centralwidget")
        window.setCentralWidget(self.centralwidget)

        self.label_raw_pic = QtWidgets.QLabel(self.centralwidget)
        self.label_raw_pic.setGeometry(QtCore.QRect(350, 50, 450, 450))
        self.label_raw_pic.setStyleSheet("background-color:#bbbbbb;")
        self.label_raw_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw_pic.setObjectName("label_raw_pic")

        self.label_aligned_face = QtWidgets.QLabel(self.centralwidget)
        self.label_aligned_face.setGeometry(QtCore.QRect(900, 50, 150, 150))
        self.label_aligned_face.setStyleSheet("background-color:#bbbbbb;")
        self.label_aligned_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_aligned_face.setObjectName("label_aligned_face")

        self.Title1 = QtWidgets.QLabel(self.centralwidget)
        self.Title1.setGeometry(QtCore.QRect(900, 200, 100, 16))
        self.Title1.setStyleSheet("font: 10pt \"Bahnschrift SemiLight SemiConde\";")
        self.Title1.setObjectName("Title1")
        self.Title1.setText("Emotion...")

        # self.label_aligned_face2 = QtWidgets.QLabel(self.centralwidget)
        # self.label_aligned_face2.setGeometry(QtCore.QRect(650, 550, 100, 100))
        # self.label_aligned_face2.setStyleSheet("background-color:#bbbbbb;")
        # self.label_aligned_face2.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_aligned_face2.setObjectName("label_aligned_face")

        # self.Title2 = QtWidgets.QLabel(self.centralwidget)
        # self.Title2.setGeometry(QtCore.QRect(650, 650, 100, 16))
        # self.Title2.setStyleSheet("font: 10pt \"Bahnschrift SemiLight SemiConde\";")
        # self.Title2.setObjectName("Title2")
        # self.Title2.setText("Emotion...")
        #
        # self.label_aligned_face3 = QtWidgets.QLabel(self.centralwidget)
        # self.label_aligned_face3.setGeometry(QtCore.QRect(800, 550, 100, 100))
        # self.label_aligned_face3.setStyleSheet("background-color:#bbbbbb;")
        # self.label_aligned_face3.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_aligned_face3.setObjectName("label_aligned_face")
        #
        # self.Title3 = QtWidgets.QLabel(self.centralwidget)
        # self.Title3.setGeometry(QtCore.QRect(800, 650, 100, 16))
        # self.Title3.setStyleSheet("font: 10pt \"Bahnschrift SemiLight SemiConde\";")
        # self.Title3.setObjectName("Title2")
        # self.Title3.setText("Emotion...")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 100, 200, 50))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.open_file_browser)

        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(50, 200, 200, 50))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.pushButton2.setFont(font)
        self.pushButton2.setObjectName("pushButton")
        self.pushButton2.clicked.connect(self.open_camera)

        self.Title4 = QtWidgets.QLabel(self.centralwidget)
        self.Title4.setGeometry(QtCore.QRect(50, 330, 300, 100))
        self.Title4.setStyleSheet("font: 10pt \"Bahnschrift SemiLight SemiConde\";")
        self.Title4.setObjectName("Title4")
        self.Title4.setText("Please choose your model below:")
        self.Title4.setStyleSheet("color: white; font-weight: bold; font:10pt \"Times New Roman\";")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(50, 400, 200, 50))
        self.comboBox.setStyleSheet("background-color: rgb(222, 222, 222);""font: 10pt \"Times New Roman\";")
        self.comboBox.setObjectName('comboBox')
        self.comboBox.addItems(['VGG16', 'VGG19'])
        self.comboBox.currentIndexChanged.connect(self.load_model)

        self.scroll_area = QtWidgets.QScrollArea(self.centralwidget)
        self.scroll_area.setGeometry(QtCore.QRect(350, 550, 450, 200))
        self.scroll_area.setStyleSheet("background-color: rgb(222, 222, 222);")
        self.scroll_area.setObjectName("scrollArea")
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        self.chart_widget = QChartView(self.centralwidget)
        self.chart_widget.setGeometry(820, 260, 380, 500)
        self.draw_bar_chart()

        self.retranslate_ui(window)
        QtCore.QMetaObject.connectSlotsByName(window)

    def show_detail(self, result):
        image = result[0]
        frame = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (100, 100))
        pred = result[1]
        score = result[2][0].cpu().tolist()
        self.draw_bar_chart(score)
        self.Title1.setText(pred)
        self.label_aligned_face.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))
        self.label_aligned_face.setScaledContents(True)

    def retranslate_ui(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("Window", "Window"))
        self.label_raw_pic.setText(_translate("Window", "Ready for input"))
        self.label_aligned_face.setText(_translate("Window", "(¬‿¬)"))
        self.pushButton.setText(_translate("Window", "Image"))
        self.pushButton2.setText(_translate("Window", "Camera"))

    def show_raw_img(self, filename):
        img = cv2.imread(filename)
        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (350, 400))
        self.label_raw_pic.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))
        self.label_raw_pic.setScaledContents(True)

    # def show_aligned_face(self, images):
    #     for index, image in enumerate(images):
    #         frame = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (100, 100))
    #         if index == 0:
    #             self.label_aligned_face1.setPixmap(QtGui.QPixmap.fromImage(
    #                 QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
    #                              QtGui.QImage.Format_RGB888)))
    #         if index == 1:
    #             self.label_aligned_face2.setPixmap(QtGui.QPixmap.fromImage(
    #                 QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
    #                              QtGui.QImage.Format_RGB888)))
    #         if index == 2:
    #             self.label_aligned_face3.setPixmap(QtGui.QPixmap.fromImage(
    #                 QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
    #                              QtGui.QImage.Format_RGB888)))

    # def show_pred_results(self, predicts):
    #     for index, predict in enumerate(predicts):
    #         if index == 0:
    #             self.Title1.setText(predict)
    #         if index == 1:
    #             self.Title2.setText(predict)
    #         if index == 2:
    #             self.Title3.setText(predict)

    def open_file_browser(self):
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="Choose an image", directory="./exp_data/",
                                                                     filter="All Files (*);;Text Files (*.txt)")
        if file_name is not None and file_name != "":
            self.show_raw_img(file_name)
            img, height, width = get_image(file_name)
            faces = detect_face(img)
            cropped_images, _ = detect_landmarks(img, faces, height, width)
            # self.show_aligned_face(cropped_images)
            # processed_images = process_img(cropped_images)

            result_list = self.get_result_list(cropped_images)
            self.show_result_list(result_list)

            # preds, scores = predicting(processed_images, self.model)
            # scores = scores[0].cpu().tolist()
            # self.show_pred_results(preds)
            # print(preds)
            # print(scores)

    def open_camera(self):
        camera(self.model)

    def load_model(self):
        use_cuda = True
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        if self.comboBox.currentText() == "VGG19":
            model = VGG19(num_classes=8)
            net = torch.load('./model/VGG19_80.t7', map_location=device)
            checkpoint = net['model']
            model.load_state_dict(checkpoint)
            print("VGG19")

        if self.comboBox.currentText() == "VGG16":
            model = VGG16(num_classes=8)
            net = torch.load('./model/VGG16_81.t7', map_location=device)
            checkpoint = net['model']
            model.load_state_dict(checkpoint)
            print("VGG16")

        self.model = model

    def get_result_list(self, cropped_images):
        result_list = {}
        processed_images = process_img(cropped_images)
        preds, scores = predicting(processed_images, self.model)

        for index, cropped_image in enumerate(cropped_images):
            result_list[index] = [cropped_image, preds[index], scores[index]]

        return result_list

    def show_result_list(self, result_list):
        while self.scroll_layout.count() > 0:
            widget_item = self.scroll_layout.itemAt(0)
            widget = widget_item.widget()
            if widget:
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()

        for i in range(len(result_list)):
            index = i + 1
            result = result_list[i]

            button = QtWidgets.QPushButton("Face NO." + str(index))
            button.clicked.connect(partial(self.show_detail, result))
            button.setFixedSize(420, 30)
            button.setStyleSheet("""
                        QPushButton {
                            background-color: lightgray;
                            border: None;
                            padding: 5px;
                        }
                        QPushButton:pressed {
                            background-color: gray;
                            border: None;
                            padding: 5px;
                        }
                    """)
            self.scroll_layout.addWidget(button)

        self.scroll_area.setWidget(self.scroll_widget)

    def draw_bar_chart(self, pred=None):
        # self.chart_widget = QChartView(self.centralwidget)
        # self.chart_widget.setGeometry(820, 260, 380, 500)  # Set the geometry for the chart

        chart = QChart()
        # chart.setTitle("Bar Chart")

        series = QHorizontalBarSeries()

        set0 = QBarSet("possibilities")
        if pred:
            values = pred
        else:
            values = [0, 0, 0, 0, 0, 0, 0, 0]
        set0.append(values)

        series.append(set0)
        chart.addSeries(series)

        axisX = QBarCategoryAxis()
        categories = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
        axisX.append(categories)
        chart.addAxis(axisX, QtCore.Qt.AlignLeft)
        series.attachAxis(axisX)

        self.chart_widget.setChart(chart)
