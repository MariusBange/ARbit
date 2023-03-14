"""This module shall serve as a toolbox to extract the position, velocity and orbital elements for any given satellite to a given time"""

import pandas as pd
import numpy as np
from sgp4.api import Satrec

# The path to the file containing the used TLEs
pathToTLEFile = None

# Data structured in the form of three element tuples, containing the name and both of the lines of the two-line-element 
setsOfTLE = []

def __init__(self, dataPath):
	if not dataPath:
		self.pathToTLEFile = "../../Data/TLE.txt"
	else:
		self.pathToTLEFile = dataPath
	initialize(pathToTLEFile)


def initialize(filename):
	"""Subroutine to initialize TLEs by filename.
	
	Keyword arguments:
	filename -- contains the name and path of the file
	
	returns:
	the now setup setsOfTLE in 3-element tuples
	"""
	if filename:
		pathToTLEFile = filename
	else:
		pathToTLEFile = "../../Data/TLE.txt"
	with open(pathToTLEFile) as TLEs:
		content = TLEs.read().splitlines()
	
	setsOfTLE = []
	for index in range(0, len(content), 3):
		setsOfTLE.append((content[index], content[index + 1], content[index + 2]))
	return setsOfTLE
		

def findTLE(satelliteID):
	"""Function to select and read a TLE.

	Keyword arguments:
	satelliteID -- shall identify the satellite by name
	
	returns:
	an array with three elements: Its ID and the other two each representing one line respectively
	"""
	global setsOfTLE
	if not setsOfTLE:
		setsOfTLE = initialize(None)

	for s in setsOfTLE:
		if satelliteID in s[0]:
			return s
	print("Satellite not found!")
	return None


def propagate(satellite, timestamp):
	"""Function to propagate the satellites current position (according to the given TLEs).
	
	Keyword arguments:
	satellite -- described in the form of its TLE
	timestamp -- represents the desired time for propagation whereas 'None' will be interpreted as now
	
	returns:
	the cartesian coordinates, velocities and timestamp representing the system state of the satellite (using SGP4)
	"""

	if not timestamp:
		timestamp = pd.Timestamp.now().to_julian_date()
	satelliteRV = Satrec.twoline2rv(satellite[1], satellite[2])
	julianDate = int(timestamp)
	fraction = timestamp % 1.0
	error, position, velocity = satelliteRV.sgp4(julianDate, fraction)
	print("Current position of satellite (in cartesian ECI-frame): ")
	print(position)
	print("Current velocity of satellite (in cartesian ECI-frame): ")
	print(velocity)
	return position, velocity, timestamp


def parseParameters(satellite):
	"""Parses the satellites orbital parameters from the TLE-format.
	
	Keyword arguments:
	satellite -- described in the form of its TLE
	
	returns:
	a set of 5 of the 6 keplerian orbital elements (excluding the true anomaly)
	"""
	
	secondLine = [var for var in satellite[2].split(" ") if var] # Don't care about revolution number or checksum, mean anomaly won't be used anyway
	
	eccentricity = float("0."+secondLine[4])
	semimajorAxis = ((3.986004418*10.**14)**(1./3.))/((float(secondLine[7])*2.*np.pi/86400.0)**(2./3.))
	inclination = float(secondLine[2])
	longitudeOfAscendingNode = np.deg2rad(float(secondLine[3]))
	argumentOfPeriapsis = np.deg2rad(float(secondLine[5]))
	meanAnomaly = float(secondLine[6]) # Not used as it is substituted by the SGP4 propagation
	keplerianElements = (eccentricity, semimajorAxis, inclination, longitudeOfAscendingNode, argumentOfPeriapsis)
	
	print("Keplerian elements (excluding true anomaly):")
	print(keplerianElements)
	return keplerianElements


def findOrbit(satelliteID, timestamp):
	"""This function shall return all the required information to display the orbit.
	
	Keyword arguments:
	satelliteID -- shall identify the satellite by name
	
	returns:
	an array of all relevant parameters for subsequent conversion and rendering
	"""
	
	satellite = findTLE(satelliteID)
	systemState = propagate(satellite, timestamp)
	orbitParams = parseParameters(satellite)
	return systemState, orbitParams