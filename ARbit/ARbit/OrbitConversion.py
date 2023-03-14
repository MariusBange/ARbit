"""This module shall serve as a toolbox to convert inbetween the relevant frames frame the orbit itself to the earth inertial to the earth fixed frame"""

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, CartesianDifferential, CartesianRepresentation
import numpy as np

EARTH_RADIUS = 6371.


def frameTransformation(systemState):
	"""Subroutine to convert from the SGP4 based earth centered inertial (ECI) coordinates to earth centered earth fixed (ECEF) coordinates.
	
	Keyword arguments:
	system state -- consists of position (first element) and velocity (second element) as cartesian coordinates from an ECI frame to an also given timestamp (third element)
	
	returns:
	the position in an earth-centered frame as relative coordinates (with earth radius as 1) and a TEME (ECI) to ITRS (ECEF) transformation matrix
	"""

	# pack systemState into a TEME (True Equator Mean Equinox, a kind of ECI frame) object
	temePosition = CartesianRepresentation(systemState[0]*u.km)
	temeVelocity = CartesianDifferential(systemState[1]*u.km/u.s)
	teme = TEME(temePosition.with_differentials(temeVelocity), obstime=Time(systemState[2], format='jd'))
	
	# convert from teme to itrs to receive eart-centered positions
	itrs = teme.transform_to(ITRS(obstime=Time(systemState[2], format='jd')))

	relativeCoordinates = np.array([itrs.earth_location.x.value, itrs.earth_location.y.value, itrs.earth_location.z.value])/EARTH_RADIUS


	temex = TEME(CartesianRepresentation([1.,0.,0.]*u.km), obstime=Time(systemState[2], format='jd'))
	temey = TEME(CartesianRepresentation([0.,1.,0.]*u.km), obstime=Time(systemState[2], format='jd'))
	temez = TEME(CartesianRepresentation([0.,0.,1.]*u.km), obstime=Time(systemState[2], format='jd'))
	itrsx =	temex.transform_to(ITRS(obstime=Time(systemState[2], format='jd')))
	itrsy =	temey.transform_to(ITRS(obstime=Time(systemState[2], format='jd')))
	itrsz =	temez.transform_to(ITRS(obstime=Time(systemState[2], format='jd')))

	temeToItrsTransformationMatrix = np.array([[itrsx.earth_location.x.value, itrsx.earth_location.y.value, itrsx.earth_location.z.value],
		  [itrsy.earth_location.x.value, itrsy.earth_location.y.value, itrsy.earth_location.z.value],
		  [itrsz.earth_location.x.value, itrsz.earth_location.y.value, itrsz.earth_location.z.value]])
	
	print("Current Location of satellite over earth")
	print(itrs.earth_location)

	return (relativeCoordinates, np.linalg.inv(temeToItrsTransformationMatrix))


def transformOrbitalPlane(keplerianElements):
	"""Subroutine to construct a rotationmatrix from the inertial earth centered plane to the orbital plane.
	
	Keyword arguments:
	keplerianElements -- the first 5 keplerian elements, of which the inclination, longitude of the ascending node and arguments of periapsis are used
	
	returns:
	the rotation matrix from the earth centered inertial to the orbital plane
	"""

	inclination = np.deg2rad(keplerianElements[2])
	longitudeOfAscendingNodeTransform = np.array([[np.cos(keplerianElements[3]),-np.sin(keplerianElements[3]),0.],
											  [np.sin(keplerianElements[3]),np.cos(keplerianElements[3]),0.],
											  [0.,0.,1.]])
	inclinationTransform = np.array([[1.,0.,0.],
								 [0.,np.cos(inclination),-np.sin(inclination)],
								 [0.,np.sin(inclination),np.cos(inclination)]])
	argumentOfPeriapsisTransform = np.array([[np.cos(keplerianElements[4]),-np.sin(keplerianElements[4]),0.],
											  [np.sin(keplerianElements[4]),np.cos(keplerianElements[4]),0.],
											  [0.,0.,1.]])
	return np.matmul(np.matmul(argumentOfPeriapsisTransform, inclinationTransform), longitudeOfAscendingNodeTransform)


def calculateEccentricOffsetTranslation(keplerianElements):
	"""Subroutine to generate a translation matrix from the focus of the orbit (earth) to the center of the orbit (focus and center diverge in case of elliptical orbits).
	
	Keyword arguments:
	keplerianElements -- the first 5 keplerian elements, of which the inclination, longitude of the ascending node and arguments of periapsis are used
	
	returns:
	the translation matrix from orbit focus (earth) to the orbits center
	"""

	return np.array([[-keplerianElements[0]*keplerianElements[1]/EARTH_RADIUS,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def convertOrbit(orbit):
	"""This function shall compute the conversion from the inertial orbital parameters and coordinates to earth (or globe) centered ones.
	
	Keyword arguments:
	orbit -- described as a tuple of the current system state, consisting of position, velocity and time, and 5 of the 6 keplerian elements
	
	returns:
	the now setup setsOfTLE in 3-element tuples
	"""

	position, frameTransform = frameTransformation(orbit[0])
	earthToOrbitPlanarTransform = transformOrbitalPlane(orbit[1])
	eccentricTransform = calculateEccentricOffsetTranslation(orbit[1])
	globeToOrbitCenterTransform = np.matmul(frameTransform, np.matmul(earthToOrbitPlanarTransform, eccentricTransform))
	return (position, globeToOrbitCenterTransform)
