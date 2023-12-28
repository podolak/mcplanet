import random
import numpy                as np
import matplotlib.pyplot    as plt

import lib.physical         as physical
import lib.monotonic        as monotonic

class MCDensity(object):
    def __init__(self, radii, densities, pressure, fixed_density):
        self._radii = radii
        self._fixed_density = fixed_density
        self._densities = densities
        self._pressure = pressure

    def radius(self):
        return self._radii[-1]

    def get_mass(self):
        return physical.compute_mass(self._radii, self._densities, self._fixed_density)
    
    def get_density(self, radius):  
        # Fixed density, so find the first shell with larger radius,
        # Return the density of that shell.
        assert(radius <= self._radii[-1])
        
        shell_num = np.where(self._radii>= radius)[0][0]
        return self._densities[shell_num]
        
        
    def get_inner_mass(self, outer_radius):
        inner_radii = self._radii[self._radii < outer_radius]
        inner_densities = self._densities[:len(inner_radii)]
        radii = np.append(inner_radii, outer_radius)
        densities = np.append(inner_densities, self.get_density(outer_radius))
        return physical.compute_mass(radii, densities, self._fixed_density) 

    def get_moment(self):
        return physical.compute_moment(self._radii, self._densities, self._fixed_density)

    def get_densities(self):
        return self._densities

    def get_radii(self):
        return self._radii

    def get_pressure(self):
        if self._pressure is None:
            self._pressure = physical.compute_pressure(self._radii, self._densities, self._fixed_density)
        return self._pressure

    def get_mass_moment_ratio(self):
        return self.get_moment()/(self.get_mass()*self.radius()*self.radius())

    def plot_densities(self, label="density"):
        plt.plot(self._radii, self._densities, label=label)
        plt.xlabel('radius (cm)')
        plt.ylabel('density (g/cc)')
        plt.legend()

    def plot_pressure(self, label="pressure"):
        plt.plot(self._radii, self.get_pressure(), label=label)
        plt.xlabel('radius (cm)')
        plt.ylabel('pressure (dyne/cm^2)')
        plt.legend()

class MCDensityFactory(object):
    def __init__(self, mass, moment_ratio, radius, shells, num_shells, outer_two_shell_ratio, smooth, fixed_density, min_density):
        # TODO:   Need to think about/ document units.
        self._mass = mass
        self._smooth = smooth
        self._fixed_density = fixed_density
        self._radius = radius
        self._moment_ratio = moment_ratio
        self._moment = moment_ratio*mass*radius*radius
        self._min_density = min_density
        
        # Can pass in units as increasing list of radii.
        # Shells are the outer radius of the shells.   Assume shell are touching.
        # num_shells ignored in this case.
        if shells is None:
            self._shells = self._create_radii(num_shells, outer_two_shell_ratio)
        else:
            self._shells = np.array(shells)

        self._num_shells = len(self._shells)

    def _create_radii(self, num_shells, outer_two_shell_ratio):
        # We have two types of shells, "main" shells, controlled by num_shells.
        # Outer two shells are split into outer_two_shell_ratio shells.
        
        #set up inner shells:
        inner = (np.array(range(num_shells))+1)[:-2]/float(num_shells)
        outer = (np.array(range(num_shells*outer_two_shell_ratio))+1)[-2*outer_two_shell_ratio:]/float(num_shells*outer_two_shell_ratio)
        return np.concatenate((inner, outer))*self._radius
        #return (np.array(range(num_shells))+1)/float(num_shells)*self._radius

    def _normalize_mass(self, model_points):
        # Assume that our current model is (x,y)
        # Where x is the outer radius (first is zero)
        # And y is the density.
        #
        # We need to normalize so we match the total mass
        
        outer = self._shells
        
        inner= np.append([0], self._shells)
        ranges = zip(inner, outer)
        mass = physical.compute_mass(self._shells, model_points, self._fixed_density)
        return model_points*self._mass/mass

    
    def _get_vol(self):
        # Assume that our current model is (x,y)
        # Where x is the outer radius (first is zero)
        # And y is the density.
        outer = self._shells[-1]
        
        # Just the volume of a sphere with the outer radius
        return 4.0/3.0 * np.pi * np.power(outer,3.0)
    
        """
        inner= np.append([0], self._shells)
        ranges = zip(inner, outer)

        #coefficients = [physical.get_mass_coefficient(rad1, rad2, self._fixed_density)
        #                for (rad1, rad2) in ranges]
        """
        return sum(coefficients)

        
    def _normalize_moment(self, model_points):
        # Assume that our current model is (x,y)
        # Where x is the outer radius (first is zero)
        # And y is the density.
        #
        # We need to normalize so we match the total mass
        outer = self._shells
        inner= np.append([0], self._shells)
        ranges = zip(inner, outer)

        #coefficients = [physical.get_moment_coefficient(rad1, rad2, self._fixed_density)
        #                for (rad1, rad2) in ranges]
        moment = physical.compute_moment(self._shells, model_points, self._fixed_density)
        #moment = np.array(coefficients).dot(model_points)
        return model_points*self._moment/moment


    def create_mass_model(self):
        # first generate random monotonic-path
        model = monotonic.get_monotonic_vals(self._shells/self._radius, self._smooth)
        norm_model = self._normalize_mass(model)
        
        # Assure minimum density
        if norm_model[-1] < self._min_density:
            m = self._mass
            k0 = norm_model[-1]
            k = self._min_density
            v = self._get_vol()
            x =  (k-k0)*m/(m-k*v)
            norm_model = (norm_model + x)*m/(m+v*x)
        return self._shells, norm_model


    def create_moment_model(self):
        # first generate random monotonic-path
        model = monotonic.get_monotonic_vals(self._shells/self._radius, self._smooth)
        return self._shells, self._normalize_moment(model)
    
    
    def create_mass_and_moment_model(self, num_samples):
        # We're going to create a model that matches both mass and moment.
        #
        # First, we'll genreate a bunch (100 by default) models and 
        # normalize them so they have the correct mass.
        #
        # Next, we'll compute the moment for each one and divide them into 
        # two groups.    "bigger_moment" and "smaller_moment".
        #
        # Next, we'll randomly choose one model from each set and find the 
        # linear interpolation that will give us a the desired moment.   
        # This interpolated model should have both the correct mass and moment.
        models = [self.create_mass_model() for _ in range(num_samples)]
        moments = [physical.compute_moment(model[0], model[1], self._fixed_density) for model in models]
        
        #print("%s models generated"%num_samples)
        #print("Largest Moment Ratio: %s"%(max(moments)/(self._mass*self._radius*self._radius)))
        #print("Smallest Moment Ratio: %s\n\n"%(min(moments)/(self._mass*self._radius*self._radius)))
        
        bigger_moment = list(filter(lambda x: x[0] >= self._moment, zip(moments, models)))
        smaller_moment = list(filter(lambda x: x[0] < self._moment, zip(moments, models)))
       
        #print("Generated %s bigger moments and %s smaller moments\n\n"%(len(bigger_moment), len(smaller_moment)))
        
        if 0 == len(bigger_moment) or 0 == len(smaller_moment):
            print("WARNING:: Did not manage to create models with moments bracketing desired result.   Exiting.")
            print("Try running again?\n\n")
            return self._shells, 0*models[0][1]
        
        bigger = random.choice(bigger_moment)
        smaller = random.choice(smaller_moment)
        alpha = (self._moment-smaller[0])/(bigger[0]-smaller[0])
        
        return MCDensity(self._shells, alpha*bigger[1][1] + (1.0-alpha)*smaller[1][1], pressure=None, fixed_density = self._fixed_density)
    
"""
Function to create default models.

@param mass: mass of the planet
@param moment_ratio:  mass/moment ratio of the planet.
@param radius: radius of the planet.
@param num_shells: number of "main shells.
@param outer_two_shell_ratio: split outer two shells into more.  (default=1, no splitting).
@param num_random_samples:  number of "raw" random models we create before trying to fit moment.   
                            Defaults to 100, but can set smaller (e.g. 10) for faster models at the expense
                            of occasionally not getting a model.
@param smooth: size of smoothing kernel.   Default is 201.   Probably not a good idea to touch.
@param fixed_density:  true if we assume fixed density throught the model (default for early papers).   
                       Continuous (piecewise linear) denisty now also mostly supported (not for pressure/temp).
@param min_density:  Can put a lower bound on the min density of outer layer.
"""
def create_mcdensity(mass, moment_ratio, radius, num_shells, outer_two_shell_ratio=1, smooth=201, fixed_density=True, min_density=0.0, num_random_samples=100):
    factory = MCDensityFactory(mass, moment_ratio, radius, None, num_shells, outer_two_shell_ratio, smooth, fixed_density, min_density)
    return factory.create_mass_and_moment_model(num_random_samples)
    