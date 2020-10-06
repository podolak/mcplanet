import physical
import monotonic
import numpy as np
import matplotlib.pyplot as plt
from mc_density import MCDensity, create_mcdensity

class MCInterior(MCDensity):
    def __init__(self, radii, densities, rock, env, fixed_density=True, use_cgs=True):
        super(MCInterior, self).__init__(radii, densities, fixed_density)
        self._rock = rock
        self._env = env
        self._cgs = use_cgs

    def get_rock_pct(self):
        return self._rock

    def get_env_pct(self):
        return self._env
        
    def plot_rock(self):
        plt.plot(self._radii, self._rock, label="rock ratio")

    def plot_env(self):
        plt.plot(self._radii, self._env, label="env ratio")
        
    def get_temp(self):
        return physical.compute_temp(self._radii, self._densities, self._rock, self._env, self._cgs)

    def plot_temp(self):
        plt.plot(self._radii, self.get_temp())


class MCInteriorFactory(object):
    def __init__(self, pct_rock, pct_env, density_model):
        self._r = pct_rock
        self._e = pct_env
        self._w = 1.0-pct_rock-pct_env
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
            
        # The amout of mass in all three is about right.   We'll the largest
        # of these and just multiplty the percentage in each shell.   The 
        # reason we take the largest is to avoid have a percent higher than 1.0.
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
    
def create_mcinterior(mass, moment_ratio, radius, pct_rock, pct_env, num_shells=100, rock_0=1.0, env_0=1.0):
    mcdensity = create_mcdensity(mass, moment_ratio, radius, num_shells)
    factory = MCInteriorFactory(pct_rock, pct_env, mcdensity)
    rock = factory.create_rock_model(rock_0)
    env = factory.create_env_model(env_0)

    # TODO:   Write a cls method as a second constructor for this.
    # Otherwise, we're recreating the MCDensity over again.
    return MCInterior(mcdensity._radii, mcdensity._densities, rock, env)




    
