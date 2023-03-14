from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from Processor import *
import OrbitInterpretation as oi
import OrbitConversion as oc

class Satellite():

    name = None
    model: 'path to .obj file' = None
    globeToOrbitCenterTransform = None#: float[3][3] = None
    eccentricity: float = None
    semimajorAxis: float = None
    position: float = None

    def __init__(self, name, model, mtx=None, dist=None):
        super().__init__()

        self.name = name
        self.model = model

        self.mtx = mtx
        self.dist = dist

        #self.object = self.load()

        self.colors = []
        self.lines = []

        #self.prepare()
        
        orbit = oi.findOrbit(name, None)
        self.eccentricity = orbit[1][0]
        self.semimajorAxis = orbit[1][1]/oc.EARTH_RADIUS/1000
        self.position, self.globeToOrbitCenterTransform = oc.convertOrbit(orbit)
        print(self.globeToOrbitCenterTransform)
        print(self.position)
    
    def load(self):
        vtx = []
        idx = []
        
        f = open(self.model)
        for l in f.readlines():
            if l[0] == "v": 
                vtx.append(l[1:].split())
            elif l[0] == "f":
                idx.append(l[1:].split())
        
        idx = np.int32(np.array(idx))
        idx -= 1
        vtx = np.float32(np.array(vtx)).reshape(-1, 1, 3)
        
        return vtx, idx

    def prepare(self):
        k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        r = np.array([[np.pi / 2, 0.0, 0.0]])
        t = np.array([[-40.0, 30.0, 100.0]])

        points = np.int32(cv2.projectPoints(self.object[0], r, t, k, np.array([-1.0, 0.0, 0.0, 0.0]))[0])

        visiblePoints = []
        for point in self.object[1]:
            c2o = self.object[0][point[0]] - np.array([[0, 800, 0]])
            normal = np.cross(self.object[0][point[1]]-self.object[0][point[0]], self.object[0][point[2]]-self.object[0][point[0]])
            d = np.dot(normal[0] / np.linalg.norm(normal[0]), np.array([0, 1, 0]))
            dot = np.dot(c2o[0]/np.linalg.norm(c2o[0]), normal[0]/np.linalg.norm(normal[0]))
            if dot > 0.2:
                visiblePoints.append(point)
                self.colors.append(-d * np.array((255, 0, 255)))

        for triangle in visiblePoints:
            self.lines.append([points[triangle[0]], points[triangle[1]], points[triangle[2]]])

    def render(self, frame, globePosition, earthToScreenProjection, earthRadius, resolution, colour, rvecs):
        frame = self.drawOrbitEllipse(frame, globePosition, earthToScreenProjection, earthRadius, resolution, colour, rvecs, globePosition)
        return frame
        #return self.insert(frame)

    def insert(self, frame):
        img = frame

        for i in range(0, len(self.lines)):
            cv2.fillConvexPoly(img, np.array(self.lines[i]), self.colors[i])
        
        return img

    def drawOrbitEllipse(self, frame, globePosition, earthToScreenRotation, earthRadius, resolution, colour, rvecs, tvecs):
        """Function to draw a given orbit in the form of dots.
	
	    Keyword arguments:
        earthToScreenProjection -- transformation from the globe to the camera
        earthRadius -- the radius of the earth relative to the camera
	    resolution -- defines the resolution of the drawn orbit, directly defines the amount of drawn points
        colour -- the colour which the drawn elements shall have
        """

        #TODOs: Set dot size in relation to distance to screen, Visualize speed (by dot colour, line length, offset?), TESTING!

        semiminorAxis = np.sqrt(1-self.eccentricity**2*self.semimajorAxis**2)
        lastimgPoint = None
        
        for i in range(0, resolution+1):
            orbitalPoint = np.array([self.semimajorAxis*np.cos(2*i*np.pi/resolution),semiminorAxis*np.sin(2*i*np.pi/resolution), 0]) #Calculates points along the ellipse
            orbitalPoint = np.dot(self.globeToOrbitCenterTransform, orbitalPoint) * earthRadius * 1.5 #Transform ellipse points to earth centered frame

            #if not self.cull(orbitalPoint, globePosition, earthToScreenRotation, earthRadius):
            #    onScreenCoord = np.dot(earthToScreenRotation,orbitalPoint) #Transform ellipse points to screen and convert into 2D
            imgpt, jac = cv2.projectPoints(orbitalPoint, rvecs, tvecs, self.mtx, self.dist)
                # Draw each point along the orbit
                #print(np.int32(onScreenCoord[:2]))
            if lastimgPoint is not None:
                frame = cv2.line(frame, np.int32(imgpt)[0,0], np.int32(lastimgPoint)[0,0], colour, 2)
            lastimgPoint = imgpt
            #frame = cv2.circle(frame, np.int32(imgpt)[0,0], 5, colour, -1)#, int(10/onScreenCoord[2]))
        return frame


    def cull(self, point, globePosition, camToGlobeRotation, earthRadius):
        """Subroutine to check whether a point should be culled because it is behind the globe.

        Keyword arguments:
        point -- the point to check if it should be culled
        camToGlobe -- transformation from the globe to the camera
        earthRadius -- the radius of the earth relative to the camera

        returns:
        True when the point is "behind" the globe and False when it is not
        """
        
        point += globePosition # Order of translation and rotation may needs to be changed
        point = np.dot(camToGlobeRotation, point)
        return point[2] < 0 and np.sqrt(point[0]**2*point[1]**2) < earthRadius



class VideoThread(QThread):
    
    changePixmapSignal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.object = None#Satellite("bunny", SatelliteChecklist.findModel())

        self.runFlag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.runFlag:
            ret, frame = cap.read()
            if ret:
                if self.object is not None:
                    frame = self.object.insert(frame)
                self.changePixmapSignal.emit(frame)
            time.sleep(0.05)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self.runFlag = False
        self.wait()



class SatelliteChecklist(QtWidgets.QDialog):
    
    satellites = ["ISS (ZARYA)", "TIROS N", "HST", "METEOR 1-29",
        "LANDSAT 7", "LANDSAT 8", "LAGEOS 1", "LAGEOS 2",
        "GOES 2", "METEOSAT 8 (MSG 1)", "METEOSAT 9 (MSG 2)",
        "SEASAT 1", "NOAA 15", "NOAA 16", "COSMOS 2411 (GLONASS)",
        "NOAA 19", "METEOR 1-29", "SPOT 6", "NOAA 18", "IRS P6",
        "MOLNIYA 2-9", "MOLNIYA 2-10", "MOLNIYA 1-29",
        "STARLINK-1007", "STARLINK-1008", "STARLINK-1009", "STARLINK-3005", "STARLINK-3004"]
    selectedSatellites: list(satellites) = []
    modelLibrary = {
        "ISS (ZARYA)": "../../Models/ISSComplete1.obj",
        "HST": "../../Models/hubble.obj",
        "STARLINK": "../../Models/Starlink_v1.obj",
        "Default": "../../Models/Starlink_v1.obj"
        }

    def __init__(self, satelliteSelection, modelLibrary, app):
        super(SatelliteChecklist, self).__init__()

        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()
        self.satellites = satelliteSelection
        self.modelLibrary = modelLibrary

        if self.satellites is not None:
            for i in range(len(self.satellites)):
                item = QtGui.QStandardItem(self.satellites[i])
                item.setCheckable(True)
                check = Qt.Checked if self.satellites[i] in self.selectedSatellites else Qt.Unchecked
                item.setCheckState(check)
                item.setSelectable(False)
                item.setEditable(False)
                self.model.appendRow(item)
            self.listView.clicked.connect(self.stateChanged)

        self.listView.setModel(self.model)

        self.app = app

        hbox = QHBoxLayout()
        hbox.addWidget(self.listView)
        hbox.addStretch(1)
        
        self.setLayout(hbox)

    def stateChanged(self, index):
        checked = self.model.item(index.row(), 0).checkState() == 2
        if checked and len(self.selectedSatellites) == 3:
            self.model.item(index.row(), 0).setCheckState(False)
            checked = False
        if checked and self.satellites[index.row()] not in self.selectedSatellites and len(self.selectedSatellites) < 3:
            self.selectedSatellites.append(Satellite(self.satellites[index.row()], self.findModel(self.satellites[index.row()]), self.app.mtx, self.app.dist))
        elif not checked and self.satellites[index.row()] in self.selectedSatellites:
            for satellite in self.selectedSatellites:
                if satellite.name == self.satellites[index.row()]:
                    self.selectedSatellites.remove(satellite)
                    break

        self.getSatelliteData() # Should now be redundant

    def getSatelliteData(self):
        print(self.selectedSatellites)
        for satellite in self.selectedSatellites:
            ##Intert Code here to get satellite data
            pass

    def findModel(self, name) -> 'Path to satellite model':
        """Find the path to the obj. file for a given satellite or the default model if not available."""
        if "STARLINK" in name: return self.modelLibrary.get("STARLINK")
        return self.modelLibrary.setdefault(name, self.modelLibrary.get("Default"))



class ControlWidget(QGroupBox):

    def __init__(self, satelliteSelection, modelLibrary, app):
        super(ControlWidget, self).__init__()
        
        self.setFixedSize(300, 720)
        self.setTitle("Menü")
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid gray;
                border-radius: 5px;
                padding: 5px
            }

            QGroupBox::title {
                subcontrol-origin: top;
                subcontrol-position: top center; /* position at the top center */
                padding: 1 5px;
                background-color: gray;
            }""")

        self.grid = QGridLayout()

        self.labelSat = QLabel("\n   Select satellites (max 3):")
        self.checklist = SatelliteChecklist(satelliteSelection, modelLibrary, app)
        
        self.scale = 1/100
        self.labelScale = QtWidgets.QLabel("   Set scale:")
        self.sliderScale = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderScale.setMinimum(100)
        self.sliderScale.setMaximum(1000)
        self.sliderScale.setValue(100)
        self.sliderScale.setTickInterval(100)
        self.sliderScale.setSingleStep(100)
        self.sliderScale.setFixedSize(120, 30)
        self.sliderScale.valueChanged.connect(self.scaleChanged)
        self.labelScaleValue = QtWidgets.QLabel(f"   1/{self.sliderScale.value()}")
        
        self.grid.addWidget(self.labelSat, 0, 0, 1, 4)
        self.grid.addWidget(self.checklist, 1, 0, 1, 4)
        self.grid.addWidget(self.labelScale, 2, 0, 1, 4)
        self.grid.addWidget(self.labelScaleValue, 3, 0, 1, 1)
        self.grid.addWidget(self.sliderScale, 3, 1, 1, 3)


        self.setLayout(self.grid)

    def satelliteChanged(self):
        print(self.comboSat.currentText())

    def scaleChanged(self):
        if self.sliderScale.value() > self.scale and self.sliderScale.value() % 100 >= 50:
            self.scale = (self.sliderScale.value() // 100 + 1) * 100
            self.labelScaleValue.setText(f"   1/{self.scale}")
        elif self.sliderScale.value() < self.scale and self.sliderScale.value() % 100 < 50:
            self.scale = (self.sliderScale.value() // 100) * 100
            self.labelScaleValue.setText(f"   1/{self.scale}")
        self.sliderScale.setValue(self.scale)
        print(self.scale)



class App(QWidget):
    
    processor = None

    def __init__(self, processor: Processor, satelliteSelection, modelLibrary):
        super().__init__()
        self.setWindowTitle("ARbit")
        #self.displayWidth = 1280
        #self.displayHeight = 960
        self.displayWidth = 640
        self.displayHeight = 480
        
        self.imageLabel = QLabel(self)
        self.imageLabel.resize(self.displayWidth, self.displayHeight)
        self.menuItem = ControlWidget(satelliteSelection, modelLibrary, self)
        self.menu = QWidget()
        self.vmenu = QVBoxLayout()
        self.vmenu.addWidget(self.menuItem)
        self.menu.setLayout(self.vmenu)

        hbox = QHBoxLayout()
        hbox.addWidget(self.menu)
        hbox.addWidget(self.imageLabel)
        self.setLayout(hbox)

        self.processor = processor

        self.mtx = processor.mtx
        self.dist = processor.dist

        self.thread = VideoThread()
        
        self.thread.changePixmapSignal.connect(self.updateImage)
        
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def updateImage(self, cvImg):
        """Updates the imageLabel with a new opencv image"""
        rvecs, tvecs = self.processor.process(cvImg) # Earthradius is still needed?
        if rvecs is not None:
            rvec = np.reshape(rvecs, -1)
            tvec = np.reshape(tvecs, -1)
            camToGlobeRotation = cv2.Rodrigues(rvecs)[0]
            for satellite in self.menuItem.checklist.selectedSatellites:
                cvImg = satellite.render(cvImg, tvecs, camToGlobeRotation, 17.5, 50, (255, 255, 0), rvecs)

        qtImg = self.convertCvQt(cvImg)
        self.imageLabel.setPixmap(qtImg)
    
    def convertCvQt(self, cvImg):
        """Convert from an opencv image to QPixmap"""
        rgbImage = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(self.displayWidth, self.displayHeight, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # satellites = ["ISS", "TIROS", "Nimbus 1", "Kosmos 44", "ESSA", "Meteor", "Numbus 4", "Landsat 1", "Nimbus 5", "Landsat 2", "GEOS 3", "Nimbus 6", "GOES-1", "Kosmos 1076", "GOES-2", "Meteosat-1", "Landsat 3", "Explorer 58", "Goes-3", "Seasat", "Nimbus 7", "Explorer 60", "Kosmos 1119", "Meteosat-2", "NOAA-7", "Landsat-4", "NOAA-8", "Landsat 5", "NOAA-9", "Geosat", "METEOR 3-01", "SPOT 1", "NOAA-10", "MOS 1a", "IRS 1A"]
    # selectedSatellites = []
    # window = App()
    # window.show()
    # sys.exit(app.exec_())
    Satellite("ISS (ZARYA)", "../../Models/ISSComplete1.obj")
