import numpy as np
import cv2
import OrbitInterpretation as oi
import OrbitConversion as oc
from interface2 import * 
from Processor import *


app = QApplication(sys.argv)
window = App(
	Processor(standalone=False),
	satelliteSelection = ["ISS (ZARYA)", "TIROS N", "HST", "METEOR 1-29",
		"LANDSAT 7", "LANDSAT 8", "LAGEOS 1", "LAGEOS 2",
		"GOES 2", "METEOSAT 8 (MSG 1)", "METEOSAT 9 (MSG 2)",
		"SEASAT 1", "NOAA 15", "NOAA 16", "COSMOS 2411 (GLONASS)",
		"NOAA 19", "METEOR 1-29", "SPOT 6", "NOAA 18", "IRS P6",
		"MOLNIYA 2-9", "MOLNIYA 2-10", "MOLNIYA 1-29",
		"STARLINK-1007", "STARLINK-1008", "STARLINK-1009", "STARLINK-3005", "STARLINK-3004"],
	modelLibrary = {
        "ISS (ZARYA)": "../../Models/ISSComplete1.obj",
        "HST": "../../Models/hubble.obj",
		"STARLINK": "../../Models/Starlink_v1.obj",
        "Default": "../../Models/Starlink_v1.obj"
        #"Default": "../../Models/bunny.obj"
        }
	)
window.show()
sys.exit(app.exec_())


# Demo of OrbitInterpretation and Orbit Conversion
"""orbit = oi.findOrbit('ISS (ZARYA)', None)
orbit2 = oi.findOrbit('MOLNIYA 2-9', None)
convertedOrbit = oc.convertOrbit(orbit)
print(convertedOrbit)
convertedOrbit2 = oc.convertOrbit(orbit2)
print(convertedOrbit2)"""