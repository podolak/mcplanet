import numpy as np
import lib.constants    as constants

def get_shell_volume(rad1, rad2):
    return 4.0/3.0 * np.pi * (np.power(rad2,3.0) - np.power(rad1, 3.0))

def get_mass_coefficient(rad1, rad2, fixed_density):
    if not fixed_density:
        raise NotImplementedError("Linear density not implmeneted yet.")
    return 4.0/3.0 * np.pi * (np.power(rad2,3.0) - np.power(rad1, 3.0))

def get_moment_coefficient(rad1, rad2, fixed_density):
    if not fixed_density:
        raise NotImplementedError("Linear density not implemented yet.")
    return 8.0/15.0 * np.pi * (np.power(rad2,5.0) - np.power(rad1, 5.0))


def compute_mass(radii, densities, fixed_density):
    # @param radii: array with outer radii.
    # @param densities: array with desities
    if not fixed_density:
        """Trapz-integrate mass from rho(r) data."""
        from scipy.integrate import trapz
        return 4*np.pi*trapz(densities*radii**2, x=radii)
    
    outer = radii
    inner= np.append([0], radii)

    coefficients = [get_mass_coefficient(rad1, rad2, fixed_density)
                    for (rad1, rad2) in zip(inner, outer)]

    return np.array(coefficients).dot(densities)

def compute_moment(radii, densities, fixed_density):
    # @param radii: array with outer radii.
    # @param densities: array with desities
    if not fixed_density:
        """Trapz-integrate mass from rho(r) data."""
        from scipy.integrate import trapz
        return 8.0/3.0*np.pi*trapz(densities*radii**4, x=radii)

    outer = radii
    inner= np.append([0], radii)

    coefficients = [get_moment_coefficient(rad1, rad2, fixed_density)
                    for (rad1, rad2) in zip(inner, outer)]

    return np.array(coefficients).dot(densities)


def compute_pressure(radii, densities, fixed_density):
    # @param radii: array with outer radii.
    # @param densities: array with desities

    #first, we need the mass at each outer radius.
    outer = radii
    inner= np.append([0], radii)

    coefficients = [get_mass_coefficient(rad1, rad2, fixed_density) 
                    for (rad1, rad2) in zip(inner, outer)]
    mass = (coefficients*densities).cumsum()

    # For now, just assume that the mass and radius are constant throughout the shell.
    dp_dr = mass*densities*constants.G/(radii*radii)
    dp= dp_dr*(outer-inner[:-1])
    return np.cumsum(dp[::-1])[::-1]
    # np.append(aa[0], aa[:-1])
