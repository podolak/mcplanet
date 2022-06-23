G = 6.6743e-8 # cgs units.
SUN_RADIUS = 6.955e10

class Planet():
    def __init__(self, mass, moment_ratio, radius):
        self.mass = mass
        self.moment_ratio = moment_ratio
        self.radius = radius
        
URANUS = Planet(8.68e28, 0.23, 2.54e9)   

