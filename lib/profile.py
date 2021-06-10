import random
import numpy             as np
import matplotlib.pyplot as plt

import lib.temperature   as temperature
import lib.mc_interior   as mc_interior
import lib.mc_density    as mc_density
#########################################################################
# Temperature Profile.
#########################################################################

# The goal of the temperature profile is to combine an mc_density with
# A temperature catalog and create an mc_interior with the following
# constraints:
#
# Fixed max temp for innermost shell.
# Monotonic decreasing temp/pressure/density/composition.
#
class TemperatureProfile(object):
    # The idea of a temperature profile is to take a TemperatureCatalog and MCDensity.
    # We use these to create all possible profiles and choose and interior that works.
    def __init__(self, temp_catalog, density_model):
        self._catalog = temp_catalog
        self._model = density_model
        
    # Diagnostic.   Print out the temperature profile
    def plot_profile(self):
        num_shells = len(self._model.get_radii())
        for comp in self._catalog._compositions:
            rock, water, env = temperature.composition_to_mix3(comp)
            mcinterior = mc_interior.MCInterior(self._model._radii, self._model._densities, np.ones(num_shells)*rock, np.ones(num_shells)*env)
            mcinterior.plot_temp(label=comp)

    # Diagnostic.   Print out the temperature profile for the rock/env curves.
    def plot_temperature_profiles(self):
        num_shells = len(self._model.get_radii())
        for comp in self._catalog._compositions:
            rock, water, env = temperature.composition_to_mix3(comp)
            mcinterior = mc_interior.MCInterior(self._model._radii, self._model._densities, np.ones(num_shells)*rock, np.ones(num_shells)*env, self._catalog)
            mcinterior.plot_temp(label=comp)
            
    # Diagnostic.   Print out a curve that shows composition assuming a temperature of temp
    def plot_temp_curve(self, temp):
        rad = self._model.get_radii()
        rho = self._model.get_densities()
        p = self._model.get_pressure()
        
        comps = []
        for i in range(len(rad)):
            comps.append(self._catalog.get_composition(temp, rho[i], p[i]))
            
        plt.plot(rad, comps, label=temp)

    def _monotonic_composition(self, max_temp):
        # So the idea here is to create an MCInterior that
        # has both monotonically decreasing temp and composition
        # as you had to the surface of the planet.
        
        rad = self._model.get_radii()
        rho = self._model.get_densities()
        p = self._model.get_pressure()
        
        comps = []
        temps = []
        cur_comp = self._catalog.get_composition(max_temp, rho[0], p[0])
        cur_temp = max_temp
        if cur_comp is None:
            cur_comp = self._catalog._compositions.max()
          
        for i in range(len(rad)):
         
            temp_for_comp = self._catalog.get_temp(cur_comp, rho[i], p[i])

            # We'll start with comp.  
            if temp_for_comp is None:
                # So the current composition is not giving a temp.  
                # We will just push a None for now and go to the next shtell.
                comps.append(None)
                temps.append(None)
                continue  

            elif temp_for_comp < cur_temp +1:
                # This is what we want.   Just use same composition.
                comps.append(cur_comp)
                temps.append(temp_for_comp)
                cur_temp = temp_for_comp

            elif temp_for_comp >= cur_temp:
                # Using the current composition yields a higher temperature.
                # Instead, find a new composition that give that temp.
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])


                if comp_for_temp is None:
                    # We are out of bounds.   Try for next shell and hope for the best.
                    comps.append(None)
                    temps.append(None)
                    continue

                # Sanity check, new comp cannot be higher.
                if (comp_for_temp > cur_comp + 0.01):
                    comps.append(None)
                    temps.append(None)
                    #import ipdb;ipdb.set_trace()
#                assert comp_for_temp <= cur_comp, "Error: composition not monontonic"

                cur_comp = comp_for_temp
                comps.append(comp_for_temp)
                temps.append(cur_temp)
            else:
                raise "Error, unexpected value, should not get here!"
    
        #return zip(rad, rho, p, comps, temps)
        return comps
    def _inverse_monotonic_composition(self, min_temp):
        # So the idea here is to create an MCInterior that
        # has both monotonically decreasing temp and composition
        # as you had to the surface of the planet.
        
        rad = self._model.get_radii()
        rho = self._model.get_densities()
        p = self._model.get_pressure()
        
        comps = []
        temps = []
        import ipdb;ipdb.set_trace()
        cur_comp = self._catalog.get_composition(min_temp, rho[-1], p[-1])
        cur_temp = min_temp
        for i in reversed(range(len(rad))):
            temp_for_comp = self._catalog.get_temp(cur_comp, rho[i], p[i])
            
            # We'll start with comp.  
            if temp_for_comp is None:
                # So the current composition is not giving a temp.  
                # We will just push a None for now and go to the next shtell.
                comps.append(None)
                temps.append(None)
                continue  
                
            elif temp_for_comp > cur_temp:
                # This is what we want.   Just use same composition.
                comps.append(cur_comp)
                temps.append(temp_for_comp)
                cur_temp = temp_for_comp
            
            elif temp_for_comp <= cur_temp:
                # Using the current composition yields a higher temperature.
                # Instead, find a new composition that give that temp.
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])
            
                
                if comp_for_temp is None:
                    # We are out of bounds.   Try for next shell and hope for the best.
                    comps.append(None)
                    temps.append(None)
                    continue
                
                # Sanity check, new comp cannot be higher.
                assert comp_for_temp >= cur_comp, "Error: composition not monontonic"
                
                cur_comp = comp_for_temp
                comps.append(comp_for_temp)
                temps.append(cur_temp)
            else:
                raise "Error, unexpected value, should not get here!"
            
        #return zip(rad, rho, p, comps, temps)
        return reversed(comps)
    
    
    def _max_composition(self, max_temp):
        # So the idea here is to create an MCInterior that
        # has both monotonically decreasing temp and composition
        # as you had to the surface of the planet.
        
        rad = self._model.get_radii()
        rho = self._model.get_densities()
        p = self._model.get_pressure()
        
        comps = []
        temps = []
        cur_comp = self._catalog.get_composition(max_temp, rho[0], p[0])
        cur_temp = max_temp
        for i in range(len(rad)):
            temp_for_comp = self._catalog.get_temp(cur_comp, rho[i], p[i])

            # We'll start with comp.  
            if temp_for_comp is None:
                # So the current composition is not giving a temp.  
                # We will just push a None for now and go to the next shtell.
                comps.append(None)
                temps.append(None)
                continue  

            elif temp_for_comp < cur_temp+1:
                # This is what we want.   Just use same composition.
                comps.append(cur_comp)
                temps.append(temp_for_comp)
                cur_temp = temp_for_comp

            elif temp_for_comp >= cur_temp:
                # Using the current composition yields a higher temperature.
                # Instead, find a new composition that give that temp.
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])


                if comp_for_temp is None:
                    # We are out of bounds.   Try for next shell and hope for the best.
                    comps.append(None)
                    temps.append(None)
                    continue

                # Sanity check, new comp cannot be higher.
                assert comp_for_temp <= cur_comp+0.01, "Error: composition not monontonic"

                cur_comp = comp_for_temp
                comps.append(comp_for_temp)
                temps.append(cur_temp)
            else:
                raise "Error, unexpected value, should not get here!"
    
        #return zip(rad, rho, p, comps, temps)
        return comps
    
    def monotonic_interior(self, max_temp, inverse=False):
     
        if inverse:
            comps = self._inverse_monotonic_composition(max_temp)
        else:
            comps = self._monotonic_composition(max_temp)

        #import ipdb;ipdb.set_trace()
        # Take the resulting composition and create MCInterior object.
        rad = self._model.get_radii()
        rho = self._model.get_densities()
       
        # We're going to default None to previous shell.
        prev = None
        mix = []
        count = 0
        for comp in comps:
            if comp is None:
                if prev is None:
                    # First shell doesn't match anything.
                    return None, None
                count = count +1
                mix.append(temperature.composition_to_mix3(prev))
            else:
                mix.append(temperature.composition_to_mix3(comp))
                prev = comp
        rock, water, env = zip(*mix)
        if count > 0.10 * len(rad) and not inverse:
            return None, None
        return mc_interior.MCInterior(rad, rho, rock, env, self._catalog), count
    
    
def get_fixed_temp_model(mass, moment_ratio, radius, num_shells, 
                         max_temp, temperature_catalog, smooth=101, 
                         inverse=False, seed=None, full_model= False):
    if seed == None:
        seed = round(random.random(),9)
    random.seed(seed)
    try:
        mcdensity = mc_density.create_mcdensity(mass, moment_ratio, radius, num_shells=num_shells, smooth=smooth)
    
        profile = TemperatureProfile(temperature_catalog, mcdensity)
        inter, count =  profile.monotonic_interior(max_temp, inverse)
    except:
        print("Unexpected error with seed = %s"%seed)
        return seed, None, None
    
    if inter is None:
        return seed, None, None
        
    if full_model:
        return seed, inter, count
    
    else:
        return seed, inter.compute_ratios(), count
