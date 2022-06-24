import numpy             as np
import matplotlib.pyplot as plt

import lib.physical      as physical
import lib.monotonic     as monotonic
import lib.temperature   as temperature
from lib.mc_density      import MCDensity, create_mcdensity

class MCInterior(MCDensity):
    def __init__(self, radii, densities, mix, catalog, fixed_density=True, use_cgs=True):
        super(MCInterior, self).__init__(radii, densities, fixed_density)
        self._mix = np.array(mix)
        self._catalog = catalog

    def get_mix(self):
        cols = self._mix.shape[1]
        return [self._mix[:,i] for i in range(cols)]
    
    def get_composition(self):
        return [self._catalog.mix_to_composition(x) for x in self._mix]
    
    def plot_mix(self):
        names = self._catalog.get_table_names()
        mix = self.get_mix()
        for name, material in zip(names, mix):
            plt.plot(self._radii, material, label=name)
        plt.xlabel('radius (cm)')
        plt.ylabel('percent material')
        plt.legend()
    
    def plot_composition(self, label="composition"):
        plt.plot(self._radii, self.get_composition(), label=label)
        plt.xlabel('radius (cm)')
        plt.ylabel('composition')
        plt.legend()
        
    def get_mix_mass(self, name):
        names = self._catalog.get_table_names()
        for i in range(len(names)):
            if names[i] == name:
                comp = self.get_mix()[i]
                return physical.compute_mass(self._radii, self._densities*comp, self._fixed_density)
                
        print ("Cannot find composition with name %s"%name)
        return None
         
    def get_mix_ratios(self):
        retval = []
        for name in self._catalog.get_table_names():
            retval.append(round(self.get_mix_mass(name)/self.get_mass(), 2))
        return retval

    def get_temp(self):
        pressures = self.get_pressure()
        temps = []
        for rho, p, mix in zip(self._densities, pressures, self._mix):
            comp = self._catalog.mix_to_composition(mix)
            temps.append(self._catalog.get_temp(comp, rho, p))
        return np.array(temps)
    
    def get_inner_temp(self):
        pressure = self.get_pressure()[0]
        mix= self._mix[0]
        density = self._densities[0]
        composition = self._catalog.mix_to_composition(mix)
        return self._catalog.get_temp(composition, density, pressure)

    def plot_temp(self, label="Temperature"):
        plt.plot(self._radii, self.get_temp(), label=label)
        plt.xlabel('radius (cm)')
        plt.ylabel('temperature (K)')
        plt.legend()
        
    def plot_composition_density(self):
        temps = self.get_temp()
        pressure = self.get_pressure()
        plt.plot(self._radii, self._densities, label="actual")
        def get_comp_density(tbl):
            rho= []
            for t,p in zip(temps, pressure):
                rho.append(tbl.get_density(t,p))
            return rho
        for t in self._catalog._tables:
            name = t._name
            plt.plot(self._radii, get_comp_density(t), label=name)
        plt.xlabel('radius (cm)')
        plt.ylabel('density (g/cc)')
        plt.legend()
            
            
    def debug_shell(self, radius=None, shell=None):
        if shell is None and radius is None:
            print("Need to specify radius or shell number")
        
        if radius is not None:
            # convert radius to shell
            r_above = np.where(self._radii > radius)[0]
            if len(r_above) > 0:
                shell = r_above[0]
            else:
                print("radius too high")
                return 0.0
        if shell >= len(self._radii):
            print("shell index too high")
        
        # Print stats:
        print("shell index is %s"%shell)
        print("radius is %s"%self._radii[shell])
        print("density is %s (%s)"%(self._densities[shell], np.log10(self._densities[shell])))
        print("pressure is %s (%s)"%(self.get_pressure()[shell], np.log10(self.get_pressure()[shell])))
        print("composition is %s"%self.get_composition()[shell])
        print("temperature is %s (%s)"%(self.get_temp()[shell], np.log10(self.get_temp()[shell])))
        for t in self._catalog._tables:
            name = t._name
            temp = t.implied_temperature(self._densities[shell], self.get_pressure()[shell])
            print("implied temp of pure %s would be %s"%(name, temp if temp is None else 10**temp))
        for t in self._catalog._tables:
            name = t._name
            print("density of pure %s would be %s"%(name, t.get_density(self.get_temp()[shell], self.get_pressure()[shell])))
        
        
# Need to rebuild this into a general factory for random composition.
# The idea would be to either generate the outer mix and move inward 
# (maybe binary split?).   Alternatively, create a single monotonic 
# composition and interpret.  Get the ratios one at a time?  Would
# That be too much smoothing?

""""
class MCInteriorFactory(object):
    def __init__(self, catalog, mix_ratio, density_model):
     
        self._model = density_model
        self._mass = density_model.get_mass()
        self._shells = density_model._radii
        self._rho = density_model._densities

    def radius(self):
        return self._shells[-1]

    def _squish(self, percent, squish_ratio, outer=False):
        if not outer:
            # Resize the model on the y-axis, so the outer
            # Shell is squish_raio of the total model.
            new_shells = self._shells*squish_ratio
            #next, resample the percentages of the new model.
            return np.interp(self._shells,new_shells, percent, right=0.0)  
        else:  
            new_shells = (self._shells-self.radius())*squish_ratio+self.radius()
            return np.interp(self._shells,new_shells, percent, left=0.0)

    def create_rock_model(self, interior = 1.0):
        # Create a model with monotonic decreasing amounts of rock.
        # 
        # interior -- The percentage of rock in the innermost shell.   
        # Note, cannot be smaller than the total percent of rock in 
        # the total model.
        assert interior >= self._r, "Impossible to have less than %s rock in the core"%self._r

        # Start with a random distribution (1,0)
        init = 0

        while init < self._r * self._mass:
            model = monotonic.get_monotonic_vals(self._shells/self.radius())*interior
            init = physical.compute_mass(self._shells, model*self._rho)

        # Now do binary search
        max_bound = 1.0
        min_bound = 0.0
        cur_bound = 0.5

        # Current plan is to do a fixed number (10?) to get close enough, 
        # then solve for the last part

        for i in range(10):
            rock = self._squish(model, cur_bound)
            rock_mass = physical.compute_mass(self._shells, rock*self._rho)
            if  rock_mass > self._r *self._mass:
                max_bound = cur_bound
            else:
                min_bound = cur_bound
            cur_bound = (max_bound+min_bound)*0.5

        # The amout of mass in max_bound, min_bound, rock_mass is about right.   
        # We'll divide by the largest of these and just multiplty the percentage in each shell.   
        # The reason we take the largest is to avoid have a percent higher than 1.0.
        rock = self._squish(model, max_bound)
        rock_mass = physical.compute_mass(self._shells, rock*self._rho)
        rock = rock*(self._r*self._mass/rock_mass)
        return rock

    def create_env_model(self, exterior = 1.0):
        # Create a model with monotonic increasing amounts of envelope
        # (as the radius increases).
        # 
        # exterior -- The percent envolope in the outmost shell.   
        # Note, cannot be smaller than the total percent of envelope in 
        # the total model.
        assert exterior >= self._e, "Impossible to have less than %s envelope in the outer shell"%self._r

        # Start with a random distribution (1,0)
        init = 0

        while init < self._r * self._mass:
            model = monotonic.get_monotonic_vals(self._shells/self.radius())[::-1]*exterior
            # reverse the order.
            init = physical.compute_mass(self._shells, model*self._rho)

        # Now do binary search
        max_bound = 1.0
        min_bound = 0.0
        cur_bound = 0.5

        # Current plan is to do a fixed number (10?) to get close enough, 
        # then solve for the last part

        for i in range(10):
            env = self._squish(model, cur_bound, outer=True)
            env_mass = physical.compute_mass(self._shells, env*self._rho)
            if  env_mass > self._e *self._mass:
                max_bound = cur_bound
            else:
                min_bound = cur_bound
            cur_bound = (max_bound+min_bound)*0.5

        # The amout of mass in all three is about right.   We'll the largest
        # of these and just multiplty the percentage in each shell.   The 
        # reason we take the largest is to avoid have a percent higher than 1.0.
        env = self._squish(model, max_bound, outer=True)
        env_mass = physical.compute_mass(self._shells, env*self._rho)
        env = env*(self._e*self._mass/env_mass)
        return env

def create_mcinterior(mass, moment_ratio, radius, pct_rock, pct_env, catalog, num_shells=100, rock_0=1.0, env_0=1.0, num_samples=100):
    mcdensity = create_mcdensity(mass, moment_ratio, radius, num_shells, num_samples)
    factory = MCInteriorFactory(pct_rock, pct_env, mcdensity)
    rock = factory.create_rock_model(rock_0)
    env = factory.create_env_model(env_0)

    # TODO:   Write a cls method as a second constructor for this.
    # Otherwise, we're recreating the MCDensity over again.
    return MCInterior(mcdensity._radii, mcdensity._densities, rock, env, catalog)

"""
