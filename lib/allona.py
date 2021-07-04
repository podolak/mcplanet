import numpy as np

from lib.mc_density           import MCDensity
from lib.mc_interior          import MCInterior
import lib.temperature        as temperature
import lib.constants          as constants

def allona_mcdensity():
    allona_data =  [i.strip('\n').split() for i in open("data/allona/out3D_U4.txt")][2:]
    #header = allona_data[0][1:]
    data = allona_data[-502:]
    data = [[float(y) for y in x[1:]] for x in data[2:]]
    radius = [x[3]*constants.SUN_RADIUS for x in data]
    rho = [x[7] for x in data]
    #p = [x[4] for x in data]
    #temp = [x[5] for x in data]
    
    allona_planet = MCDensity(np.array(radius[1:]), np.array(rho[1:]) )
    return allona_planet

def allona_env_pct():
    allona_z =  [i.strip('\n').split() for i in open("data/allona/outmod3D_U4_final.txt")][5:]
    data = [[float(y) for y in x[1:]] for x in allona_z]
    env = [1.0-x[6] for x in data]
    return env

def allona_density():
    allona_data =  [i.strip('\n').split() for i in open("data/allona/out3D_U4.txt")][2:]
    data = allona_data[-502:]
    data = [[float(y) for y in x[1:]] for x in data[2:]]
    density = [x[7] for x in data]
    return density

def allona_temp():
    allona_data =  [i.strip('\n').split() for i in open("data/allona/out3D_U4.txt")][2:]
    data = allona_data[-502:]
    data = [[float(y) for y in x[1:]] for x in data[2:]]
    temp = [x[5] for x in data]
    return temp
    
def allona_pressure():
    allona_data =  [i.strip('\n').split() for i in open("data/allona/out3D_U4.txt")][2:]
    #header = allona_data[0][1:]
    data = allona_data[-502:]
    data = [[float(y) for y in x[1:]] for x in data[2:]]
    p = [x[4] for x in data]
    return p


def allona_mcinterior(catalog):
    mix = []
    allona_planet = allona_mcdensity()
    pressure = allona_planet.get_pressure()
    densities = allona_planet.get_densities()
    temp = allona_temp()
    
    for i in range(len(pressure)):
        comp = catalog.get_composition(temp[i], densities[i], pressure[i])
        if comp is None:
            print(i, temp[i],",",densities[i],",", pressure[i])
            mix.append(catalog.compmosition_to_mix(catalog._compositions[-1]))
            continue
        mix.append(catalog.composition_to_mix(comp))
   
    allona_interior = MCInterior(allona_planet.get_radii(), allona_planet.get_densities(), mix, catalog)
    return allona_interior