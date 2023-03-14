from PyQt5.QtWidgets import QOpenGLWidget
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *
from PIL import Image
import numpy as np
import yaml
import OrbitInterpretation as oi
import OrbitConversion as oc

class Orbit:

    def __init__(self, name, mtx, dist):
        orbit = oi.findOrbit(name, None)
        self.eccentricity = orbit[1][0]
        self.semimajorAxis = orbit[1][1]/oc.EARTH_RADIUS/1000
        self.position, self.globeToOrbitCenterTransform = oc.convertOrbit(orbit)
        self.mtx = mtx
        self.dist = dist
        self.create3DOrbit()

    def create3DOrbit(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        #glColor(0, 0, 255)

        glLineWidth(3.)
        glBegin(GL_LINE_STRIP)
        semiminorAxis = np.sqrt(1-self.eccentricity**2*self.semimajorAxis**2)
        scale = 1
        resolution = 50
        colour = (255, 255, 0)
        lastimgPoint = None
        earthRadius = 17.5 * scale
        
        for i in range(0, resolution+1):
            orbitalPoint = np.array([self.semimajorAxis*np.cos(2*i*np.pi/resolution),semiminorAxis*np.sin(2*i*np.pi/resolution), 0])  * earthRadius * 1.1#Calculates points along the ellipse
            orbitalPoint = np.dot(self.globeToOrbitCenterTransform, orbitalPoint) #Transform ellipse points to earth centered frame
            glVertex3f(orbitalPoint[0], orbitalPoint[1], orbitalPoint[2])
            #imgpt, jac = cv2.projectPoints(orbitalPoint, rvecs, tvecs, self.mtx, None)
        #glVertex3f(0, 0, 0)
        #glVertex3f(20, 0, 0)
        glEnd()
        glEndList()

    def render(self, frame, rvecs, tvecs, scale = 1, resolution = 50, colour = (255, 255, 0)):
        frame = self.drawOrbitEllipse(frame, 17.5 * scale, resolution, colour, rvecs, tvecs)
        return frame
        #return self.insert(frame)

    def drawOrbitEllipse(self, frame, earthRadius, resolution, colour, rvecs, tvecs):
        semiminorAxis = np.sqrt(1-self.eccentricity**2*self.semimajorAxis**2)
        lastimgPoint = None
        
        for i in range(0, resolution+1):
            orbitalPoint = np.array([self.semimajorAxis*np.cos(2*i*np.pi/resolution),semiminorAxis*np.sin(2*i*np.pi/resolution), 0]) #Calculates points along the ellipse
            orbitalPoint = np.dot(self.globeToOrbitCenterTransform, orbitalPoint) * earthRadius#Transform ellipse points to earth centered frame

            imgpt, jac = cv2.projectPoints(orbitalPoint, rvecs, tvecs, self.mtx, None)#self.dist)
            if lastimgPoint is not None:
                frame = cv2.line(frame, np.int32(imgpt)[0,0], np.int32(lastimgPoint)[0,0], colour, 2)
            lastimgPoint = imgpt
        return frame

class PyQtOpenGL(QOpenGLWidget):
    
    def __init__(self, processor):
        super().__init__()

        self.processor = processor
        

        self.INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])

        # self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.webcam = cv2.VideoCapture(0)

        # initialise shapes
        #self.file = None
        self.files = {}
        self.selectedSatellites = []

        # initialise texture
        self.texture_background = None

        self.mtx = processor.mtx
        self.dist = processor.dist

        self.orbits = {}
        
        self.scale = 1.1

        #self.orbit = Orbit("ISS (ZARYA)", self.mtx, self.dist)

        #self.cam_matrix,self.dist_coefs,rvecs,tvecs = self.getCamMatrix("camera_matrix_aruco.yaml")

    def getCamMatrix(self,file):
        with open(file) as f:
            # loadeddict = yaml.safe_load(f)
            loadeddict = yaml.load(f)
            cam_matrix = np.array(loadeddict.get('camera_matrix'))
            dist_coeff = np.array(loadeddict.get('dist_coeff'))
            rvecs = np.array(loadeddict.get('rvecs'))
            tvecs = np.array(loadeddict.get('tvecs'))
            return cam_matrix,dist_coeff,rvecs,tvecs

    def initializeGL(self): #Gets called once before the first time resizeGL() or paintGL() is called.
        glClearColor(0.0, 0.0, 0.0, 0.0) #black
        glClear(GL_COLOR_BUFFER_BIT)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)

        projection_matrix = np.zeros((4,4))
        for i in range(0,3):
            for j in range(0,3):
                print(self.mtx[i,j])

        zf = 1000
        zn = 0.1

        projection_matrix[0,0] = self.mtx[0,0] / self.mtx[0,2]
        projection_matrix[1,1] = self.mtx[1,1] / self.mtx[1,2]
        projection_matrix[2,2] = - (zf + zn)/(zf-zn)
        projection_matrix[3,2] = - (2*zf*zn)/(zf-zn)
        projection_matrix[2,3] = -1
        print(projection_matrix)
        self.globe = gluNewQuadric()

        # projection_matrix[0,0] = 2 * self.mtx[0,0] / 960
        # projection_matrix[1,1] = self.mtx[1,1] / self.mtx[1,2]
        # projection_matrix[2,2] = - (zf + zn)/(zf-zn)
        # projection_matrix[3,2] = - (2*zf*zn)/(zf-zn)
        # projection_matrix[2,3] = -1
        # print(projection_matrix)

        backGroundZ = 999
        self.backGroundCoordinates = (  (-backGroundZ/projection_matrix[0,0], -backGroundZ/projection_matrix[1,1], -backGroundZ),
                                        (backGroundZ/projection_matrix[0,0], -backGroundZ/projection_matrix[1,1], -backGroundZ),
                                        (backGroundZ/projection_matrix[0,0], backGroundZ/projection_matrix[1,1], -backGroundZ),
                                        (-backGroundZ/projection_matrix[0,0], backGroundZ/projection_matrix[1,1], -backGroundZ))

        glLoadMatrixf(projection_matrix)
        # gluPerspective(37, 1.3, 0.1, 100.0) #https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
        # gluPerspective(45, self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)/self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT), 0.1, 1000.0)
        # gluPerspective(45, self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)/self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

        glLightfv(GL_LIGHT0, GL_POSITION,  (0, 0, 0, 1)) #0, 0, 0, 1 #-40, 300, 200, 0.0
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.8, 0.8, 0.8, 1.0)) #0.0, 0.0, 0.0, 1.0 #0.2, 0.2, 0.2, 1.0
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0)) #1.0, 1.0, 1.0, 1.0 #0.5, 0.5, 0.5, 1.0
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        
        # Load 3d object
        default = r'..\..\Models\clementine_v01.obj'
        #default = '../../Models/clementine/clementine_v01.obj'
        iss = r'..\..\Models\ISSComplete1.obj'
        cube = r'..\..\Models\Cube.obj'
        hubble = r'..\..\Models\hubble3.obj'
        starlink = r'..\..\Models\Starlink.obj'
        #Files = [default, iss, cube, hubble, starlink]

        self.files["DEFAULT"] = OBJ(default, 2,swapyz=True)
        self.files["ISS (ZARYA)"] = OBJ(iss,30,swapyz=True)
        self.files["HST"] = OBJ(hubble,0.1,swapyz=True)
        self.files["STARLINK"] = OBJ(starlink,0.3,swapyz=True)

        # Load 3d object
        # File = 'clementine_v01.obj'
        #File = '../../Models/ISSComplete1.obj'
        # File = '../../Models/Cube.obj'
        # File = 'hubble.obj'
        # File = 'Starlink_v1'

        #self.file = OBJ(File,swapyz=True)
 
        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

    def paintGL(self): #Gets called whenever the widget needs to be updated.
        
        # get image from webcam
        _, image = self.webcam.read()
        
        if image is not None:
            #image = cv2.undistort(image, self.mtx, self.dist)
            for sat in self.selectedSatellites:
                if sat not in self.orbits:
                    self.orbits[sat] = Orbit(sat, self.mtx, self.dist)
            rvecs, tvecs = self.processor.process(image)
            image = cv2.undistort(image, self.mtx, self.dist)
            # if rvecs is not None:
            #     for sat in self.selectedSatellites:
            #         image = self.orbits[sat].render(image, rvecs, tvecs, scale = self.scale)
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
    
            # convert image to OpenGL texture format
            bg_image = cv2.flip(image, 0)
            bg_image = Image.fromarray(bg_image)     
            ix = bg_image.size[0]
            iy = bg_image.size[1]
            bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

    
            # create background texture
            glBindTexture(GL_TEXTURE_2D, self.texture_background)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image) #https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
            
            # draw background
            glBindTexture(GL_TEXTURE_2D, self.texture_background)
            glPushMatrix()
            #glTranslatef(0.0,0.0,10.0) #letzte Variable verschiebt Hintergrund nach vorne/hinten: je kleiner desto weiter weg/hinten
            self.drawBackground()
            glPopMatrix()
            self.showSatellites(image, rvecs, tvecs)
    
    def drawBackground(self):
        # draw background
        glBegin(GL_QUADS)
        # glTexCoord2f(0.0, 1.0); glVertex3f(-840.8, -431.1, -999)#0.9999) #https://docs.microsoft.com/en-us/windows/win32/opengl/glvertex3f
        # glTexCoord2f(1.0, 1.0); glVertex3f( 840.8, -431.1, -999)#0.9999)
        # glTexCoord2f(1.0, 0.0); glVertex3f( 840.8,  431.1, -999)#0.9999)
        # glTexCoord2f(0.0, 0.0); glVertex3f(-840.8,  431.1, -999)#0.9999)

        glTexCoord2f(0.0, 1.0); glVertex3f(self.backGroundCoordinates[0][0], self.backGroundCoordinates[0][1], self.backGroundCoordinates[0][2])#0.9999) #https://docs.microsoft.com/en-us/windows/win32/opengl/glvertex3f
        glTexCoord2f(1.0, 1.0); glVertex3f(self.backGroundCoordinates[1][0], self.backGroundCoordinates[1][1], self.backGroundCoordinates[1][2])#0.9999)
        glTexCoord2f(1.0, 0.0); glVertex3f(self.backGroundCoordinates[2][0], self.backGroundCoordinates[2][1], self.backGroundCoordinates[2][2])#0.9999)
        glTexCoord2f(0.0, 0.0); glVertex3f(self.backGroundCoordinates[3][0], self.backGroundCoordinates[3][1], self.backGroundCoordinates[3][2])#0.9999)

        glEnd()
 
    def showSatellites(self, image, rvecs, tvecs):
        
        # view_matrix = np.array([(1., 0., 0., 0.),(0., 1., 0., 0.),(1., 1., -1., 8.),(0., 0., 0., 1.)])
        view_matrix = np.array([(1., 0., 0., 0.),(0., 1., 0., 0.),(0., 0., 1., 0.),(0., 0., 0., -100.)])

        if rvecs is not None:
            rmtx = cv2.Rodrigues(rvecs)[0]
            tvec = np.reshape(tvecs, -1)
            #print(tvec)
            view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvec[0]],
                                    [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvec[1]],
                                    [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvec[2]],
                                    [0.0       ,0.0       ,0.0       ,1.0    ]])

            # view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],0.5],
            #                         [rmtx[1][0],rmtx[1][1],rmtx[1][2],0],
            #                         [rmtx[2][0],rmtx[2][1],rmtx[2][2],0],
            #                         [0.0       ,0.0       ,0.0       ,1.0    ]])

            # view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
            #                         [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
            #                         [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
            #                         [0.0       ,0.0       ,0.0       ,1.0    ]])

        glColorMask(False, False, False, False)
        glPushMatrix()
        view_matrix_tmp = view_matrix * self.INVERSE_MATRIX
        view_matrix_tmp = np.transpose(view_matrix_tmp)
        glLoadMatrixd(view_matrix_tmp)
        gluSphere(self.globe, 17.5, 10, 10)
        glPopMatrix()
        glColorMask(True, True, True, True)

        #for f in self.files:
        for satellite in self.selectedSatellites:
            
            scale = np.eye(4, 4)
            scale[0,0] = self.scale
            scale[1,1] = self.scale
            scale[2,2] = self.scale
            view_matrix_tmp = np.dot(view_matrix, scale)
            glPushMatrix()
            view_matrix_tmp = view_matrix_tmp * self.INVERSE_MATRIX
            view_matrix_tmp = np.transpose(view_matrix_tmp)
            glLoadMatrixd(view_matrix_tmp)
            glCallList(self.orbits[satellite].gl_list)
            glPopMatrix()

            trans = np.eye(4, 4)
            satPos = self.orbits[satellite].position
            trans[0,0] = self.scale
            trans[1,1] = self.scale
            trans[2,2] = self.scale
            trans[0, 3] = satPos[0] * 17.5 * self.scale * 1.1
            trans[1, 3] = satPos[1] * 17.5 * self.scale * 1.1
            trans[2, 3] = satPos[2] * 17.5 * self.scale * 1.1
            view_matrix_tmp = np.dot(view_matrix, trans)

            view_matrix_tmp = view_matrix_tmp * self.INVERSE_MATRIX

            view_matrix_tmp = np.transpose(view_matrix_tmp)

            # load view matrix and draw shape
            glPushMatrix()
            #view_matrix = view_matrix
            glLoadMatrixd(view_matrix_tmp)
            if "STARLINK" in satellite:
                glCallList(self.files["STARLINK"].gl_list)
            elif satellite == "ISS (ZARYA)":
                glCallList(self.files["ISS (ZARYA)"].gl_list)
            elif satellite == "HST":
                glCallList(self.files["HST"].gl_list)
            else:
                glCallList(self.files["DEFAULT"].gl_list)
            #glCallList(self.files[f].gl_list)
            glPopMatrix()
        self.update()
