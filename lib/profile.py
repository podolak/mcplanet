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
    def plot_temperature_profile(self):
        num_shells = len(self._model.get_radii())
        for comp, name in zip(self._catalog._compositions, self._catalog.get_table_names()):
            mix = self._catalog.composition_to_mix(comp)
            mcinterior = mc_interior.MCInterior(self._model._radii, self._model._densities, np.outer(np.ones(num_shells), mix), self._catalog)
            mcinterior.plot_temp(label=name)
        plt.legend()
            
    # Diagnostic.   Print out a curve that shows composition assuming a temperature of temp
    def plot_temp_curve(self, temp):
        rad = self._model.get_radii()
        rho = self._model.get_densities()
        p = self._model.get_pressure()
        
        comps = []
        for i in range(len(rad)):
            comps.append(self._catalog.get_composition(temp, rho[i], p[i]))
            
        plt.plot(rad, comps, label=temp)
        plt.legend()

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
            cur_comp = self._catalog.get_max_comp(rho[0], p[0])
          
        for i in range(len(rad)):
         
            temp_for_comp = self._catalog.get_temp(cur_comp, rho[i], p[i])

            # We'll start with comp.  
            if temp_for_comp is None:
                # Try comp_for_temp
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])
                if comp_for_temp is None or comp_for_temp > cur_comp + 0.01:
                    # So the current composition is not giving a temp.  
                    # We will just push a None for now and go to the next shtell.
                    comps.append(None)
                    temps.append(None)
                    continue  
                else:
                    cur_comp = comp_for_temp
                    comps.append(cur_comp)
                    temps.append(cur_temp)
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
        cur_comp = self._catalog.get_composition(min_temp, rho[-1], p[-1])
        cur_temp = min_temp
        if cur_comp is None:
            cur_comp = self._catalog._compositions.min()
        for i in reversed(range(len(rad))):
            #print(rho[i], p[i])
            #print(cur_temp, cur_comp)
            #import ipdb;ipdb.set_trace()
            # special case:
            if rho[i] == 0 and p[i] == 0:
                comps.append(self._catalog._compositions.min())
                temps.append(0)
                continue
            temp_for_comp = self._catalog.get_temp(cur_comp, rho[i], p[i])
            
            # We'll start with comp.  
            if temp_for_comp is None:
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])
                if comp_for_temp is None:
                    # Everything is None.  Start a new batch
                    new_comp = self._catalog.get_min_comp(rho[i], p[i])
                    new_temp = self._catalog.get_temp(new_comp, rho[i], p[i])
                    if new_temp is None or new_comp is None:
                        #import ipdb;ipdb.set_trace()
                        comps.append(None)
                        temps.append(None)
                        continue  
                    if new_temp > cur_temp -1 and new_comp > cur_comp - 0.01:
                        cur_comp = new_comp
                        cur_temp = new_temp
                        comps.append(cur_comp)
                        temps.append(cur_temp)
                        continue
                    else:
                        comps.append(None)
                        temps.append(None)
                        continue  
                    
                if comp_for_temp > cur_comp - 0.01:
                    cur_comp = comp_for_temp
                    comps.append(cur_comp)
                    temps.append(cur_temp)
                    continue
                
                # So the current composition is not giving a temp.  
                # We will just push a None for now and go to the next shtell.
                comps.append(None)
                temps.append(None)
                continue  
                
            elif temp_for_comp > cur_temp -1:
                # This is what we want.   Just use same composition.
                comps.append(cur_comp)
                temps.append(temp_for_comp)
                cur_temp = temp_for_comp
                continue
            
            elif temp_for_comp <= cur_temp:
                # Using the current composition yields a lower temperature.
                # Instead, find a new composition that give that temp.
                comp_for_temp = self._catalog.get_composition(cur_temp, rho[i], p[i])
            
                
                if comp_for_temp is None:
                    # We are out of bounds.   Try for next shell and hope for the best.
                    #import ipdb;ipdb.set_trace()
                    comps.append(None)
                    temps.append(None)
                    continue
                
                # Sanity check, new comp cannot be higher.
                assert comp_for_temp >= cur_comp - 0.01, "Error: composition not monontonic"
                
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
                    #import ipdb;ipdb.set_trace()
                    return None, None
                count = count +1
                mix.append(self._catalog.composition_to_mix(prev))
            else:
                mix.append(self._catalog.composition_to_mix(comp))
                prev = comp
                
        if count > 0.10 * len(rad):
            return None, None
        return mc_interior.MCInterior(rad, rho,  mix, self._catalog), count
    
    
def get_fixed_temp_model(mass, moment_ratio, radius, num_shells, 
                         max_temp, temperature_catalog, smooth=101, 
                         inverse=False, seed=None, full_model= False):
    if seed == None:
        seed = round(random.random(),9)
    random.seed(seed)

    
    mcdensity = mc_density.create_mcdensity(mass, moment_ratio, radius, num_shells=num_shells, smooth=smooth)

    profile = TemperatureProfile(temperature_catalog, mcdensity)
    inter, count =  profile.monotonic_interior(max_temp, inverse)
   
   
    
    if inter is None:
        return seed, None, None, None
    
    inner_temp = inter.get_inner_temp()
    if full_model:
        return seed, inter, inner_temp, count
    
    else:
        return seed, inter.get_mix_ratios(), inner_temp, count
