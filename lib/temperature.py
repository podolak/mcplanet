import numpy as np

#########################################################################
# Temperature Computation.
#########################################################################

# The goal of the temperature module is to deal with the following relation:
# composition = f(temperature, pressure, density)
# temp = f(composition, pressure, density)
#
# As such, the main two functions for this are:
# def get_temp(comp, p, rho)
# def get_comp(temp, p, rho)
#
# In the future we may want to produce p or rho, but leaving it for now.


MIN_LOG_VAL = 0.0001 # For log(temp), log(pressure) purposes.


# Interpolate some mixture composition.
# Assuming equal temp/pressure, the rule for density is:
#
# rho(rho1, rho2, mix1, mix2) = 1.0/(mix1/rho1 + mix2/rho2)
def interpolate_composition(density, d_above, d_below):
    # Need to find the right mix
    desired_val = 1.0/density
    m1 = 1.0/d_above
    m2 = 1.0/d_below
    
    # desired_val = a*m1 + (1-a)*m2 = a*m1 + m2 -a*m2
    # desired_val - m2 = a * (m1-m2)
    # (desired_val - m2)/(m1-m2) = a
    return (desired_val-m2)/(m1-m2)
#########################################################################
# TemperatureTable
#########################################################################
# A TemperatureTable is a cached version of one of the density+pressure->temperature files.
# We currently store ANEOS and AQUA files.

class TemperatureTable(object):
    def __init__(self, comp, name, temp, pressure, density):
        # @param filename - the filename of the table.
        # @param comp - the composition value of the table
        # @param use_log - geometric interpolation if true.
        self._comp = comp
        self._name = name
        self._temp = np.log10(temp)
        self._density = np.log10(density)
        self._pressure = np.log10(pressure)
        
    def get_name(self):
        return self._name

    def _get_bracketing_temperatures_idx(self, temp):
        t_below = np.where(self._temp < temp)[0]
        if len(t_below) > 0:
            t_below = t_below[-1]
        else:
            # Temperature lower than anything on table.
            t_below = None

        t_above = np.where(self._temp >= temp)[0]
        if len(t_above) > 0:
            t_above = t_above[0]
        else:
            # Density too high.
            t_above = None
            
        return t_below, t_above

    
    def _get_bracketing_pressures_idx(self, temp_idx, pressure):
        p_below = np.where(np.logical_and(self._pressure < pressure, self._temp == self._temp[temp_idx]))[0]
        if len(p_below) > 0:
            p_below = p_below[-1]
        else:
            # Pressure too low.
            p_below = None
    
        p_above = np.where(np.logical_and(self._pressure >= pressure, self._temp == self._temp[temp_idx]))[0]
        if len(p_above) > 0:
            p_above = p_above[0]
        else:
            # Pressure too high.
            p_above = None
        
        return p_below, p_above
        
    
    def get_density(self, temperature, pressure, debug=False):
        temperature = np.log10(max(temperature, MIN_LOG_VAL))
        pressure = np.log10(max(pressure, MIN_LOG_VAL)) 
        
        # In the past, I've tried using scipy.interpolate.griddata.   
        # It didn't quite work for me for two reasons.   
        # 1.   The points are not on a regular grid, and the results I got were noisy.
        # 2.   I want to return None when the denisty/pressure are out of bounds of the grid.   scipy didn't let me do it cleanly.
        #
        # As such, we're going to our own do simple bilinear interpolation.   
         
        # First find location of density boundary.
        t1_idx, t2_idx = self._get_bracketing_temperatures_idx(temperature)
        # No density, no temp.
        if debug:
            print(t1_idx, t2_idx )
        if t1_idx is None or t2_idx is None:
            return None

        # For each density, find below, above pressures.
        t1p1_idx, t1p2_idx = self._get_bracketing_pressures_idx(t1_idx, pressure)
        t2p1_idx, t2p2_idx = self._get_bracketing_pressures_idx(t2_idx, pressure)
        
        if debug:
            print(t1p1_idx, t1p2_idx )
            print(t2p1_idx, t2p2_idx )
        # If pressure is out of bounds, no temp.
        if t1p1_idx is None or t1p2_idx is None or t2p1_idx is None or t2p2_idx is None: 
            return None
        
        # Bilinear interpolation
        rval = self._density[t1p1_idx] * (self._temp[t2_idx] - temperature)/(self._temp[t2_idx] - self._temp[t1_idx]) * (self._pressure[t1p2_idx] - pressure)/(self._pressure[t1p2_idx] - self._pressure[t1p1_idx]) + \
               self._density[t1p2_idx] * (self._temp[t2_idx] - temperature)/(self._temp[t2_idx] - self._temp[t1_idx]) * (pressure - self._pressure[t1p1_idx])/(self._pressure[t1p2_idx] - self._pressure[t1p1_idx]) + \
               self._density[t2p1_idx] * (temperature - self._temp[t1_idx])/(self._temp[t2_idx] - self._temp[t1_idx]) * (self._pressure[t2p2_idx] - pressure)/(self._pressure[t2p2_idx] - self._pressure[t2p1_idx]) + \
               self._density[t2p2_idx] * (temperature - self._temp[t1_idx])/(self._temp[t2_idx] - self._temp[t1_idx]) * (pressure - self._pressure[t2p1_idx])/(self._pressure[t2p2_idx] - self._pressure[t2p1_idx])
        return 10**rval


    def max_log_temp(self, pressure, num_steps=20):
        # First try max temp
        if self.get_density(10**self._temp.max(), pressure) is not None:
            return self._temp.max()

        # Find a point that is not none
        step = self._temp.max()-self._temp.min()
        for i in range(1, num_steps):
            cur_temp = self._temp.max()-step*i
            if cur_temp is None:
                continue

            else:
                # Use binary search to find true minimum
                max_temp = cur_temp + step
                min_temp = cur_temp
                for i in range(20):
                    cur_temp = 0.5 * (min_temp+max_temp)
                    cur_rho = self.get_density(10**cur_temp, pressure)
                    if cur_rho is None:
                        max_temp = cur_temp
                    else:
                        min_temp = cur_temp
                return min_temp
        return None

    def min_log_temp(self, pressure, num_steps=20):
        # First try min temp
        if self.get_density(10**self._temp.min(), pressure) is not None:
            return self._temp.min()

        # Find a point that is not none
        step = self._temp.max()-self._temp.min()
        for i in range(1, num_steps):
            cur_temp = self._temp.min()+step*i
            if cur_temp is None:
                continue

            else:
                # Use binary search to find true minimum
                max_temp = cur_temp
                min_temp = cur_temp - step
                for i in range(20):
                    cur_temp = 0.5 * (min_temp+max_temp)
                    cur_rho = self.get_density(10**cur_temp, pressure)
                    if cur_rho is None:
                        min_temp = cur_temp
                    else:
                        max_temp = cur_temp
                return max_temp
        return None

    def implied_temperature(self, density, pressure, force=False, debug=False):
        lowest_temp = self.min_log_temp(max(pressure, MIN_LOG_VAL))
        highest_temp = self.max_log_temp(max(pressure, MIN_LOG_VAL))
        low_log_temp = lowest_temp
        high_log_temp = highest_temp
        for i in range(20):
            cur_temp = 0.5*(low_log_temp+high_log_temp) 
            cur_rho = self.get_density(10**cur_temp, pressure)
            if debug:
                print(cur_temp, cur_rho)
            if cur_rho < density:
                high_log_temp = cur_temp
            else:
                low_log_temp = cur_temp
        if abs(density-self.get_density(10**cur_temp, pressure)) > 0.01 and not force:
            return None
        return cur_temp

#########################################################################
# TemperatureTable
#########################################################################
class TemperatureCatalog(object):
    def __init__(self, name, tables):
        self._name = name
        self._compositions = np.array([tt._comp for tt in tables])
        self._names = np.array([tt._name for tt in tables])
        self._tables = tables

        # Just make sure things are sorted.
        assert((self._compositions == sorted(self._compositions)).all())

    def get_table_names(self):
        return self._names
    
    def mix_to_composition(self, mix):
        assert(len(self._compositions) == len(mix))
        return (self._compositions * np.array(mix)).sum()
    
    def composition_to_mix(self, composition):
        retval = self._compositions*0.0
        c_below = np.where(self._compositions <= composition)[0]
        if len(c_below) > 0:
            c_below = c_below[-1]
        else:
            c_below = None
        c_above = np.where(self._compositions >= composition)[0]
        if len(c_above) > 0:
            c_above = c_above[0]
        else:
            c_above = None
        if c_below == c_above:
            retval[c_below] = 1.0
        else:
            retval[c_above] = composition - self._compositions[c_below]
            retval[c_below] = self._compositions[c_above] - composition
        return retval
    
    
    def get_min_comp(self, density, pressure):
        # Minimum temperature for any composition:
        max_comp = self._compositions[-1]
        min_comp = self._compositions[0]
        for i in range(20):
            cur_comp = 0.5*(max_comp + min_comp)
            if self.get_temp(cur_comp, density, pressure) is None:
                min_comp = cur_comp
            else:
                max_comp = cur_comp
        return max_comp
    
    def get_max_comp(self, density, pressure):
        # Minimum temperature for any composition:
        max_comp = self._compositions[-1]
        min_comp = self._compositions[0]
        for i in range(20):
            cur_comp = 0.5*(max_comp + min_comp)
            if self.get_temp(cur_comp, density, pressure) is None:
                max_comp = cur_comp
            else:
                min_comp = cur_comp
        return max_comp
        
    def get_temp(self, composition, density, pressure, debug=False):
        try:
            c_below = np.where(self._compositions <= composition)[0]
        except:
            import ipdb;ipdb.set_trace()
        if len(c_below) > 0:
            c_below = c_below[-1]
        else:
            c_below = None

        c_above = np.where(self._compositions >= composition)[0]
        if len(c_above) > 0:
            c_above = c_above[0]
        else:
            c_above = None

        # Note that it is technically possible to have the same table twice.   
        # This is not a bug -- we should get the right answer.
            
        if c_above is None or c_below is None:
            composition = None
        t_above = self._tables[c_above]
        t_below = self._tables[c_below]        
        if pressure == 0:
            return 0
        try:
            min_temp = t_below.implied_temperature(density, pressure, True)
            max_temp = t_above.implied_temperature(density, pressure, True)
        except:
            return None
        if min_temp == max_temp:
            if t_below.implied_temperature(density, pressure) is not None:
                return 10**max_temp
            else:
                # The only reason we got the same temperature is because we set forec=true.   
                # It's not the real temp.
                return None
        if debug:
            print("temp ranges are: %s %s"%(min_temp, max_temp))
        if min_temp is None or max_temp is None:
            return None
        for i in range(20):
            cur_temp = 0.5*(min_temp+max_temp) 
            cur_comp = self.get_composition(10**cur_temp, density, pressure)
            if debug:
                print("i, temp, comp = %s %s %s"%(i, cur_temp, cur_comp))
            if cur_comp is None:
                # Need to decide if to go "up" or "down".
                # One of the two solutions should be non-null.   Go back in that direction and stop.
                # This is the closest we get.
                d =  self.get_composition(10**min_temp, density, pressure)
                u = self.get_composition(10**max_temp, density, pressure)
                if d is None and u is None:
                    return None
                elif d is None:
                    cur_temp = max_temp
                    break;
                else:
                    cur_temp = min_temp
                    break;
            if cur_comp > composition:
                max_temp = cur_temp
            else:
                min_temp = cur_temp
                
        if abs(composition-self.get_composition(10**cur_temp, density, pressure)) > 0.1:
            return None
        
        return 10**cur_temp
        
    def get_composition(self, temp, density, pressure, debug=False):
            densities = [t.get_density(temp, pressure) for t in self._tables]
            densities = np.array([x for x in map(lambda x:-1 if x is None else x, densities)])
            d_below = np.where(np.logical_and(densities < density, densities >= 0))[0]
            if len(d_below) > 0:
                d_below = d_below[-1]
            else:
                d_below = None

            d_above = np.where(densities >= density)[0]
            if len(d_above) > 0:
                d_above = d_above[0]
            else:
                d_above = None

            if d_above is None or d_below is None:
                composition = None
            
            else:
                mix =  interpolate_composition(density, densities[d_above], densities[d_below])
                composition = self._compositions[d_below] + (self._compositions[d_above]-self._compositions[d_below])*mix
            if debug:
                return composition, densities
            return composition

        
#########################################################################
# Global Catalog Cache
#########################################################################
temperature_catalog_cache = {}

def temp_pressure_to_density_table(filename, name, comp):
    data = [i.strip('\n').split(',') for i in open(filename)][2:]
    t = [float(x[0]) for x in data]
    p = [float(x[1]) for x in data]
    rho = [float(x[2]) for x in data]
    return TemperatureTable(comp, name, 10**np.array(t), 10**np.array(p), 10**np.array(rho))


def sio2_density_table():
     return temp_pressure_to_density_table("data/SiO2_temp_pressure_to_density.txt", "SiO2", 1.0)
    
def iron_density_table():
     return temp_pressure_to_density_table("data/Fe_temp_pressure_to_density.txt", "iron", 2.0)
    
def dunite_density_table():
    return temp_pressure_to_density_table("data/dunite_temp_pressure_to_density.txt", "dunite", 1.0)

def water_density_table():
    return temp_pressure_to_density_table("data/water_temp_pressure_to_density.txt", "water", 0.0)

def env_density_table():
    return temp_pressure_to_density_table("data/env_temp_pressure_to_density.txt", "env", -1.0)

def allona_env_density_table():
    return temp_pressure_to_density_table("data/allona_env_temp_pressure_to_density.txt", "allona_env", -1.0)

def Z_density_table():
    # Z is 65% percent dunite, 35%% water.
    return temp_pressure_to_density_table("data/Z_temp_pressure_to_density.txt", "allona_Z", 0.0)
    
def sio2_water_2_1_density_table():
    return temp_pressure_to_density_table("data/sio2_water_2_1_temp_pressure_to_density.txt", "2:1 rock:water", 0.0)

def sio2_water_3_1_density_table():
    return temp_pressure_to_density_table("data/sio2_water_3_1_temp_pressure_to_density.txt", "3:1 rock:water", 0.0)

def sio2_water_4_1_density_table():
    return temp_pressure_to_density_table("data/sio2_water_4_1_temp_pressure_to_density.txt", "4:1 rock:water", 0.0)
    
def build_catalog(catalog_name, table_list):
    global temperature_catalog_cache
    if not catalog_name in temperature_catalog_cache:
        temperature_catalog_cache[catalog_name] = TemperatureCatalog(catalog_name, table_list)
    return temperature_catalog_cache[catalog_name]
        
def Z_env_catalog():
    return build_catalog("mix_env_catalog", [env_density_table(), Z_density_table(), dunite_density_table()])

def sio2_water_env_catalog():
    return build_catalog("sio2_water_env_catalog", [env_density_table(), water_density_table(), sio2_density_table()])

def non_log_sio2_water_env_catalog():
    return build_catalog("non_log_sio2_water_env_catalog", [env_density_table(), non_log_water_density_table(), non_log_sio2_density_table()])

def dunite_water_env_catalog():
    return build_catalog("dunite_water_env_catalog", [env_density_table(), water_density_table(), dunite_density_table()])

def iron_sio2_water_env_catalog():
    return build_catalog("iron_dunite_water_env_catalog", [env_density_table(), water_density_table(), sio2_density_table(), iron_density_table()])

def allona_model_catalog():
    return build_catalog("allona_model_catalog", [allona_env_density_table(), sio2_water_2_1_density_table(), sio2_density_table()])

def sio2_water_2_1_catalog():
    return build_catalog("sio2_water_2_1_model_catalog", [env_density_table(), sio2_water_2_1_density_table(), sio2_density_table()])

def sio2_water_3_1_catalog():
    return build_catalog("sio2_water_3_1_catalog", [env_density_table(), sio2_water_3_1_density_table(), sio2_density_table()])

def sio2_water_4_1_catalog():
    return build_catalog("sio2_water_4_1_catalog", [env_density_table(), sio2_water_4_1_density_table(), sio2_density_table()])
