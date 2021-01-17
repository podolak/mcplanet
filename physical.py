import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import math

G = 6.6743e-8 # cgs units.

def plot_model(radii, value, label = None):
    plt.plot(radii, value, label=label)
    if label is not None:
        plt.legend()
        
def get_mass_coefficient(rad1, rad2, fixed_density=True):
    if not fixed_density:
        raise NotImplemented("Linear density not implmeneted yet.")
    return 4.0/3.0 * np.pi * (np.power(rad2,3.0) - np.power(rad1, 3.0))
        
        
def get_moment_coefficient(rad1, rad2, fixed_density=True):
    if not fixed_density:
        raise NotImplemented("Linear density not implemented yet.")
    return 8.0/15.0 * np.pi * (np.power(rad2,5.0) - np.power(rad1, 5.0))


def compute_mass(radii, densities, fixed_density=True):
    # @param radii: array with outer radii.
    # @param densities: array with desities
    outer = radii
    inner= np.append([0], radii)
  
    coefficients = [get_mass_coefficient(rad1, rad2, fixed_density) 
                    for (rad1, rad2) in zip(inner, outer)]
        
    return np.array(coefficients).dot(densities)

def compute_moment(radii, densities, fixed_density=True):
    # @param radii: array with outer radii.
    # @param densities: array with desities
    outer = radii
    inner= np.append([0], radii)
        
    coefficients = [get_moment_coefficient(rad1, rad2, fixed_density) 
                    for (rad1, rad2) in zip(inner, outer)]
        
    return np.array(coefficients).dot(densities)


def compute_pressure(radii, densities, fixed_density=True, use_cgs=True):
    # @param radii: array with outer radii.
    # @param densities: array with desities

    if not use_cgs:
        raise NotImplemented("Currently only cgs units supported.")
    
    #first, we need the mass at each outer radius.
    outer = radii
    inner= np.append([0], radii)
      
    coefficients = [get_mass_coefficient(rad1, rad2, fixed_density) 
                    for (rad1, rad2) in zip(inner, outer)]
    mass = (coefficients*densities).cumsum()
   
    # For now, just assume that the mass and radius are constant throughout the shell.
    dp_dr = mass*densities*G/(radii*radii)
    dp= dp_dr*(outer-inner[:-1])
    return np.cumsum(dp[::-1])[::-1]

#########################################################################
# Temperature Computation.
#########################################################################
    
def load_temp_table(filename):
    data = [i.strip('\n').split('\t') for i in open(filename)][3:]
    density = [float(x[0]) for x in data]
    pressure = [float(x[2]) for x in data]
    temp = [float(x[3]) for x in data]
    return density, pressure, temp


# Global tables.   Read once and return results.
rock_tables = {}
env_tables = {}
def cache_temp_tables():
    global rock_tables, env_tables
    if len(rock_tables) == 0:
        for i in range(11):
            pct_rock = 10*i
            pct_water = 100-pct_rock
            filename = "temp_tables/ANEOS.%sdunite%swater.table"%(pct_rock, pct_water)
            
            d, p,  t = load_temp_table(filename)
            rock_tables[i*10] = np.array([d, p,t])
        
        for i in range(11):
            pct_env = 10*i
            pct_water = 100-pct_env
            filename = "temp_tables/ANEOS.%swater%sh_he.table"%(pct_water, pct_env)
            d, p, t = load_temp_table(filename)
            env_tables[i*10] = np.array([d,p,t])
    return rock_tables, env_tables




def _interpolate(table, density, pressure):
    try:
        log_d = np.log(density)
        log_p = np.log(pressure)

        # Split up
        d_table, p_table, t_table = table
        d_table = np.log(d_table)
        p_table = np.log(p_table)
        # First find location of density boundary.
        d_below = np.where(d_table < log_d)[0]
        if len(d_below) > 0:
            d_below = d_below[-1]
        else:
            d_below = np.argmin(d_table)
            
        d_above = np.where(d_table > log_d)[0]
        if len(d_above) > 0:
            d_above = d_above[0]
        else:
            d_above = np.argmax(d_table)
  
        d1 = d_table[d_below]
        d2 = d_table[d_above]

        # Do the two below
        range_below = np.where(d_table == d1)
        db_p_below = np.where(p_table[range_below] < log_p)[0]
        if len(db_p_below) > 0:
            db_p_below=db_p_below[-1]
        else:
            db_p_below = np.argmin(p_table[range_below])            
        db_p_above = np.where(p_table[range_below] > log_p)[0]
        if len(db_p_above) > 0:
            db_p_above=db_p_above[0]
        else:
            db_p_above = np.argmax(p_table[range_below])
            
        p1 = p_table[range_below][db_p_below]
        p2 = p_table[range_below][db_p_above]
        t1 = t_table[range_below][db_p_below]
        t2 = t_table[range_below][db_p_above]
        if p1 == p2:
            t_below = t1
        else:
            t_below = t1+ (t2-t1)*(log_p-p1)/(p2-p1)

        # Do the two above
        range_above = np.where(d_table == d2)
        da_p_below = np.where(p_table[range_above] < log_p)[0]
        if len(da_p_below) > 0:
            da_p_below=da_p_below[-1]
        else:
            da_p_below = np.argmin(p_table[range_above])    
            
        da_p_above = np.where(p_table[range_above] > log_p)[0]
        if len(da_p_above) > 0:
            da_p_above=da_p_above[0]
        else:
            db_p_above = np.argmax(p_table[range_above])
            
        p1 = p_table[range_above][da_p_below]
        p2 = p_table[range_above][da_p_above]
        t1 = t_table[range_above][da_p_below]
        t2 = t_table[range_above][da_p_above]
        if p1==p2:
            t_above= t1
        else:
            t_above = t1+ (t2-t1)*(log_p-p1)/(p2-p1)
    except:
        import ipdb;ipdb.set_trace()
    if d1==d2:
        return t_below
    return t_below + (t_above-t_below)*(log_d-d1)/(d2-d1)
    
    

def _rock_water_temp(percent_rock, density, pressure):
    rock, env = cache_temp_tables()
    # get the values from above and below
    below = math.floor(percent_rock*10.0)*10.0
    above = below+10
    val1 = _interpolate(rock[int(below)], density, pressure)
    if below >= 100:
        return val1
    val2 = _interpolate(rock[int(above)], density, pressure)
    return (val1 * (above-percent_rock*100.0) + \
            val2 * (percent_rock*100.0-below))/10.0

def _water_env_temp(percent_env, density, pressure):
    rock, env = cache_temp_tables()
    # get the values from above and below
    below = math.floor(percent_env*10.0)*10.0
    above = below+10
    val1 = _interpolate(env[int(below)], density, pressure)
    if below >= 100:
        return val1
    val2 = _interpolate(env[int(above)], density, pressure)
    return (val1 * (above-percent_env*100.0) + \
            val2 * (percent_env*100.0-below))/10.0

def _old_rock_water_temp(percent_rock, density, pressure):
    rock, env = cache_temp_tables()
    # get the values from above and below
    below = math.floor(percent_rock*10.0)*10.0
    above = below+10
    d,p,t = rock[int(below)]
    val1  = scipy.interpolate.griddata((np.log(d),np.log(p)),t,(np.log(density), np.log(pressure)), 'nearest')
    if below >= 100:
        return val1
    
    d,p,t = rock[int(above)]
    val2  = scipy.interpolate.griddata((np.log(d),np.log(p)),t,(np.log(density), np.log(pressure)), 'nearest')
  
    return (val1 * (above-percent_rock*100.0) + \
            val2 * (percent_rock*100.0-below))/10.0
    
    
def _old_water_env_temp(percent_env, density, pressure):
    rock, env = cache_temp_tables()
    # get the values from above and below
    below = math.floor(percent_env*10.0)*10.0
    above = min(100, below+10)
    d,p,t = env[int(below)]
    val1  = scipy.interpolate.griddata((np.log(d),np.log(p)),t,(np.log(density), np.log(pressure)), 'nearest')
    if below >= 100:
        return val1
    
    d,p,t = env[int(above)]
    val2  = scipy.interpolate.griddata((np.log(d),np.log(p)),t,(np.log(density), np.log(pressure)), 'nearest')
  
    return (val1 * (above-percent_env*100.0) + \
            val2 * (percent_env*100.0-below))/10.0
    
def compute_temp(radii, densities, rock, env, use_cgs= True):
    # do this sloppy.
    pressure = compute_pressure(radii, densities)
    # need to convert to mks for tables.
    pressure = pressure *0.1
    
    temp = []
    for rad, rho, prs, r_pct, e_pct in zip(radii, densities, pressure, rock, env):
        if r_pct > 0:
            # density to mks
            temp.append(_rock_water_temp(r_pct, 1000*rho, prs))
        else:
            temp.append(_water_env_temp(e_pct, 1000*rho, prs))
    return np.array(temp)
    
def compute_old_temp(radii, densities, rock, env, use_cgs= True):
    # do this sloppy.
    pressure = compute_pressure(radii, densities)
    # need to convert to mks for tables.
    pressure = pressure *0.1
  
    temp = []
    for rad, rho, prs, r_pct, e_pct in zip(radii, densities, pressure, rock, env):
        if r_pct > 0:
            # density to mks
            temp.append(_old_rock_water_temp(r_pct, 1000*rho, prs))
        else:
            temp.append(_old_water_env_temp(e_pct, 1000*rho, prs))
    return np.array(temp)
