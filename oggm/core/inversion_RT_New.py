"""Glacier thickness.


Note for later: the current code is oriented towards a consistent framework
for flowline modelling. The major direction shift happens at the
flowlines.width_correction step where the widths are computed to follow the
altitude area distribution. This must not and should not be the case when
the actual objective is to provide a glacier thickness map. For this,
the geometrical width and some other criteria such as e.g. "all altitudes
*in the current subcatchment* above the considered cross-section are
contributing to the flux" might give more interpretable results.

References:

Farinotti, D., Huss, M., Bauder, A., Funk, M. and Truffer, M.: A method to
    estimate the ice volume and ice-thickness distribution of alpine glaciers,
    J. Glaciol., 55(191), 422-430, doi:10.3189/002214309788816759, 2009.

Huss, M. and Farinotti, D.: Distributed ice thickness and volume of all
    glaciers around the globe, J. Geophys. Res. Earth Surf., 117(4), F04010,
    doi:10.1029/2012JF002523, 2012.

Bahr  Pfeffer, W. T., Kaser, G., D. B.: Glacier volume estimation as an
    ill-posed boundary value problem, Cryosph. Discuss. Cryosph. Discuss.,
    6(6), 5405-5420, doi:10.5194/tcd-6-5405-2012, 2012.

Adhikari, S., Marshall, J. S.: Parameterization of lateral drag in flowline
    models of glacier dynamics, Journal of Glaciology, 58(212), 1119-1132.
    doi:10.3189/2012JoG12J018, 2012.
"""
# Built ins
import logging
import warnings
import os
import pickle
import sys
import traceback
import pdb


# External libs
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy import optimize

# Locals
from oggm import utils, cfg
from oggm import entity_task
from oggm.core.gis import gaussian_blur
from oggm.core.massbalance import ConstantMassBalance
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.cfg import G
from oggm import workflow
from oggm import tasks

# PyGEM
#import pygem_input as pygem_prms

# Module logger
log = logging.getLogger(__name__)

# arbitrary constant
MIN_WIDTH_FOR_INV = 10

prms_from_reg_priors=False
prms_from_glac_cal=True

@entity_task(log, writes=['inversion_input'])
def prepare_for_inversion(gdir, add_debug_var=False,
                          invert_with_rectangular=True,
                          invert_all_rectangular=False,
                          invert_with_trapezoid=True,
                          invert_all_trapezoid=False,
                          filesuffix=''):
    """Prepares the data needed for the inversion.

    Mostly the mass flux and slope angle, the rest (width, height) was already
    computed. It is then stored in a list of dicts in order to be faster.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_debug_var : bool
        add some debug variables to the inversion input
    invert_with_rectangular : bool
        if False, the inversion will not consider rectangular flowlines
    invert_all_rectangular : bool
        if True, all flowlines will be considered rectangular
    invert_with_trapezoid : bool
        if False, the inversion will not consider trapezoidal flowlines
    invert_all_trapezoid : bool
        if True, all flowlines will be considered trapezoidal
    filesuffix : str
        add a suffix to the output file
    """

    # variables
    fls = gdir.read_pickle('inversion_flowlines', filesuffix=filesuffix)

    towrite = []
    for fl in fls:

        # Distance between two points
        dx = fl.dx * gdir.grid.dx

        # Widths
        widths = fl.widths * gdir.grid.dx

        # Heights
        hgt = fl.surface_h
        angle = -np.gradient(hgt, dx)  # beware the minus sign

        # Flux needs to be in [m3 s-1] (*ice* velocity * surface)
        # fl.flux is given in kg m-2 yr-1, rho in kg m-3, so this should be it:
        rho = cfg.PARAMS['ice_density']
        
        # This might error if usuer didnt compute apparent MB
        try:
            flux = fl.flux * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / rho
            # unit of flux here is m3 s-1
        except TypeError:
            raise InvalidWorkflowError('Flux through flowline unknown. '
                                       'Did you compute the apparent MB?')
        flux_out = fl.flux_out * (gdir.grid.dx**2) / cfg.SEC_IN_YEAR / rho
        # unit of flux_out here is m3 s-1
        # Clip flux to 0
        if np.any(flux < -0.1):
            log.info('(%s) has negative flux somewhere', gdir.rgi_id)
            print('(%s) has negative flux somewhere', gdir.rgi_id)
        utils.clip_min(flux, 0, out=flux)

        if np.sum(flux <= 0) > 1 and len(fls) == 1:
            log.warning("More than one grid point has zero or "
                        "negative flux: this should not happen.")
            print("More than one grid point has zero or negative flux: this should not happen.")

        if fl.flows_to is None and gdir.inversion_calving_rate == 0:
            if not np.allclose(flux_out, 0., atol=0.1):
                # TODO: this test doesn't seem meaningful here
                msg = ('({}) flux at terminus should be zero, but is: '
                       '{.4f} m3 ice s-1'.format(gdir.rgi_id, flux_out))
                raise RuntimeError(msg)

        # Shape
        is_rectangular = fl.is_rectangular
        if not invert_with_rectangular:
            is_rectangular[:] = False
        if invert_all_rectangular:
            is_rectangular[:] = True

        # Trapezoid is new - might not be available
        is_trapezoid = getattr(fl, 'is_trapezoid', None)
        if is_trapezoid is None:
            is_trapezoid = fl.is_rectangular * False
        if not invert_with_trapezoid:
            is_rectangular[:] = False
        if invert_all_trapezoid:
            is_trapezoid[:] = True

        # Optimisation: we need to compute this term of a0 only once
        flux_a0 = np.where(is_rectangular, 1, 1.5)
        flux_a0 *= flux / widths

        # Add to output
        cl_dic = dict(dx=dx, flux_a0=flux_a0, width=widths,
                      slope_angle=angle, is_rectangular=is_rectangular,
                      is_trapezoid=is_trapezoid, flux=flux,
                      is_last=fl.flows_to is None, hgt=hgt,
                      invert_with_trapezoid=invert_with_trapezoid)
        towrite.append(cl_dic)

    # Write out
    gdir.write_pickle(towrite, 'inversion_input',filesuffix=filesuffix)
    print("prepare_for_inversion is successful")


def _inversion_poly(a3, a0):
    """Solve for degree 5 polynomial with coefficients a5=1, a3, a0."""
    sols = np.roots([1., 0., a3, 0., 0., a0])
    test = (np.isreal(sols)*np.greater(sols, [0]*len(sols)))
    return sols[test][0].real


def _inversion_simple(a3, a0):
    """Solve for degree 5 polynomial with coefficients a5=1, a3=0., a0."""

    return (-a0)**(1./5.)


def _compute_thick(a0s, a3, flux_a0, shape_factor, _inv_function):
    """Content of the original inner loop of the mass-conservation inversion.

    Put here to avoid code duplication.

    Parameters
    ----------
    a0s
    a3
    flux_a0
    shape_factor
    _inv_function

    Returns
    -------
    the thickness
    """

    a0s = a0s / (shape_factor ** 3)
    a3s = a3 / (shape_factor ** 3) # Not sure whether this is correct...
    print("shape_factor in _compute_thick is :",shape_factor)
    if np.any(~np.isfinite(a0s)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    # Solve the polynomials
    try:
        out_thick = np.zeros(len(a0s))
        for i,(a0,a3,Q) in enumerate(zip(a0s,a3s, flux_a0)):
            #print("the", i, "iteration in the _compute_thick")
            out_thick[i] = _inv_function(a3, a0) if Q > 0 else 0
    except TypeError:
        # Scalar
        # print("*********************TypeError in _compute_thick*********************")
        out_thick = _inv_function(a3, a0s) if flux_a0 > 0 else 0

    if np.any(~np.isfinite(out_thick)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    return out_thick


def sia_thickness_via_optim(slope, width, flux, rel_h=1, a_factor=1,shape='rectangular',
                            glen_a=None, fs=None, t_lambda=None):
    """Compute the thickness numerically instead of analytically.

    It's the only way that works for trapezoid shapes.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx)
    width : section width in m
    flux : mass flux in m3 s-1
    rel_h : inverse of relativ heigth above buoancy
    a_factor : stress factor related to hydrostatic force at terminus
    shape : 'rectangular', 'trapezoid' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    t_lambda: the trapezoid lambda, defaults to PARAMS

    Returns
    -------
    the ice thickness (in m)
    """

    if len(np.atleast_1d(slope)) > 1:
        shape = utils.tolist(shape, len(slope))
        t_lambda = utils.tolist(t_lambda, len(slope))
        out = []
        for sl, w, f, s, t in zip(slope, width, flux, shape, t_lambda):
            out.append(sia_thickness_via_optim(sl, w, f, shape=s,
                                               glen_a=glen_a, fs=fs,
                                               t_lambda=t))
        return np.asarray(out)

    # Sanity
    if flux <= 0:
        return 0
    if width <= MIN_WIDTH_FOR_INV:
        return 0

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']
    if shape not in ['parabolic', 'rectangular', 'trapezoid']:
        raise InvalidParamsError('shape must be `parabolic`, `trapezoid` '
                                 'or `rectangular`, not: {}'.format(shape))

    # Ice flow params
    n = cfg.PARAMS['glen_n']
    fd = 2 / (n+2) * glen_a
    rho = cfg.PARAMS['ice_density']
    rhogh = (rho * cfg.G * slope) ** n

    # To avoid geometrical inconsistencies
    max_h = width / t_lambda if shape == 'trapezoid' else 1e4
    # TODO: add shape factors for lateral drag
    def to_minimize(h):
        #u = (h ** (n + 1)) * fd * rhogh + (h ** (n - 1)) * fs * rhogh
        u_drag = (rho * cfg.G * slope * h * a_factor)**n * h * fd
        u_slide = (rho * cfg.G * slope * a_factor)**n * fs * h**(n-1) * rel_h
        u = u_drag + u_slide   

        if shape == 'parabolic':
            sect = 2./3. * width * h
        elif shape == 'trapezoid':
            w0m = width - t_lambda * h
            sect = (width + w0m) / 2 * h
        else:
            sect = width * h
        return sect * u - flux
    out_h, r = optimize.brentq(to_minimize, 0, max_h, full_output=True)
    return out_h


def sia_thickness(slope, width, flux, rel_h=1, a_factor=1, shape='rectangular',
                  glen_a=None, fs=None, shape_factor=None):
    """Computes the ice thickness from mass-conservation.

    This is a utility function tested against the true OGGM inversion
    function. Useful for teaching and inversion with calving.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx) (we don't clip for min slope!)
    width : section width in m
    flux : mass flux in m3 s-1
    rel_h : inverse of relativ heigth above buoancy
    a_factor : stress factor related to hydrostatic force at terminus
    shape : 'rectangular' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    shape_factor: for lateral drag

    Returns
    -------
    the ice thickness (in m)
    """

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if shape not in ['parabolic', 'rectangular']:
        raise InvalidParamsError('shape must be `parabolic` or `rectangular`, '
                                 'not: {}'.format(shape))

    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    rho = cfg.PARAMS['ice_density']

    # Convert the flux to m2 s-1 (averaged to represent the sections center)
    flux_a0 = 1 if shape == 'rectangular' else 1.5
    flux_a0 *= flux / width

    # With numerically small widths this creates very high thicknesses
    try:
        flux_a0[width < MIN_WIDTH_FOR_INV] = 0
    except TypeError:
        if width < MIN_WIDTH_FOR_INV:
            flux_a0 = 0

    # Polynomial factors (a5 = 1)
    a0 = - flux_a0 / ((rho * cfg.G * slope*a_factor) ** 3 * fd)
    a3 = (fs / fd)*rel_h

    # Inversion with shape factors?
    sf_func = None
    if shape_factor == 'Adhikari' or shape_factor == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif shape_factor == 'Huss':
        sf_func = utils.shape_factor_huss

    sf = np.ones(slope.shape)  # Default shape factor is 1
    if sf_func is not None:

        # Start iteration for shape factor with first guess of 1
        i = 0
        sf_diff = np.ones(slope.shape)

        # Some hard-coded factors here
        sf_tol = 1e-2
        max_sf_iter = 20

        while i < max_sf_iter and np.any(sf_diff > sf_tol):
            out_thick = _compute_thick(a0, a3, flux_a0, sf, _inv_function)
            is_rectangular = np.repeat(shape == 'rectangular', len(width))
            sf_diff[:] = sf[:]
            sf = sf_func(width, out_thick, is_rectangular)
            sf_diff = sf_diff - sf
            i += 1

        log.info('Shape factor {:s} used, took {:d} iterations for '
                 'convergence.'.format(shape_factor, i))

    return _compute_thick(a0, a3, flux_a0, sf, _inv_function)


def find_sia_flux_from_thickness(slope, width, thick, glen_a=None, fs=None,
                                 shape='rectangular'):
    """Find the ice flux produced by a given thickness and slope.

    This can be done analytically but I'm lazy and use optimisation instead.
    """

    def to_minimize(x):
        h = sia_thickness(slope, width, x[0], glen_a=glen_a, fs=fs,
                          shape=shape)
        return (thick - h)**2

    out = optimize.minimize(to_minimize, [1], bounds=((0, 1e12),))
    flux = out['x'][0]

    # Sanity check
    minimum = to_minimize([flux])
    if minimum > 1:
        warnings.warn('We did not find a proper flux for this thickness',
                      RuntimeWarning)
    return flux


def _vol_below_water(surface_h, bed_h, bed_shape, thick, widths,
                     is_rectangular, is_trapezoid, fac, t_lambda,
                     dx, water_level):
    bsl = (bed_h < water_level) & (thick > 0)
    n_thick = np.copy(thick)
    n_thick[~bsl] = 0
    n_thick[bsl] = utils.clip_max(surface_h[bsl], water_level) - bed_h[bsl]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        n_w = np.sqrt(4 * n_thick / bed_shape)
    n_w[is_rectangular] = widths[is_rectangular]
    out = fac * n_thick * n_w * dx
    # Trap
    it = is_trapezoid
    out[it] = (n_w[it] + n_w[it] - t_lambda*n_thick[it]) / 2*n_thick[it]*dx
    return out


def fa_sermeq_speed_law_inv(gdir=None,mb_model=None,  mb_years=None, last_above_wl=None, v_scaling=1, verbose=False,
                     tau0=1.5, variable_yield='variable', mu=0.01,trim_profile=1,modelprms = None,glacier_rgi_table = None,
                     hindcast = None, debug = None, debug_refreeze = None, option_areaconstant = None,
                     inversion_filter = None,water_depth = None, water_level = None,filesuffix=''):
    """
    This function is used to calculate frontal ablation given ice speed forcing,
    for lake-terminating and tidewater glaciers

    @author: Ruitang Yang & Lizz Ultee

    Authors: Ruitang Yang & Lizz Ultee
    Parameters
    ----------
    gdir : py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use,here is PyGEMMassBalance
#    model : oggm.core.flowline.FlowlineModel
#        the model instance calling the function
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    flowline : oggm.core.flowline.Flowline
        the instance of the flowline object on which the calving law is called
    fl_id : float, optional
        the index of the flowline in the fls array (might be ignored by some MB models)
    last_above_wl : int
        the index of the last pixel above water (in case you need to know
        where it is).
    v_scaling: float
        velocity scaling factor, >0, default is 1
    Terminus_mb : array
        Mass balance along the flowline or nearest the terminus [m/a]. Default None, the unit meter of ice per year 
    verbose: Boolean, optional
        Whether to print component parts for inspection.  Default False.

    tau0: float, optional
        This glacier's yield strength [Pa]. Default is 150 kPa., default value here is 1.5
    variable_yield: yield_type: str, optional, variable, 'constant' or 'variable' (Mohr-Coulomb) yielding. Default is constant.
        default is variable
    mu: float, optional
        Mohr-Coulomb cohesion, a coefficient between 0 and 1. Default is 0.01.
        Only used if we have variable yield

    trim_profile: int, optional
        How many grid cells at the end of the profile to ignore.  Default is 1.
        If the initial profile is set by k-calving (as in testing) there can be a
        weird cliff shape with very thin final grid point and large velocity gradient

    water_depth: water_depth at the calving front (m)
    thick_T : thickness at the calving front (m)
    
    # parameters for the mb_model (Here is for  PyGEMMassBalance)
     ----------
    modelprms : dict
        Model parameters dictionary (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
    glacier_rgi_table : pd.Series
        Table of glacier's RGI information
    option_areaconstant : Boolean
        option to keep glacier area constant (default False allows glacier area to change annually)
    frontalablation_k : float
        frontal ablation parameter
    debug : Boolean
        option to turn on print statements for development or debugging of code
    debug_refreeze : Boolean
        option to turn on print statements for development/debugging of refreezing code
    hindcast : Boolean
        switch to run the model in reverse or not (may be irrelevant after converting to OGGM's setup)       

    Returns
    -------
    fa_viscoplastic: float
        Frontal ablation rate [m/a] based on viscoplastic assumptions
    SQFA: dict
        Frontal ablation rate [m/a] based on viscoplastic assumptions
        serface elevation at the terminus [m a.s.l.] based on OGGM ## TODO CHANGE IT BY PyGEM Surface mass balance result
        bed elevation at the terminus [m a.s.l.]
        Terminus Thickness [m]
        Yield terminus thickness [m]
        Velocity at the terminus [m/a]
        Surface mass balance at the terminus [m/a]
        Length change at the terminus [m/a] based on viscoplastic assumptions
        TODO: output the length change in case we
         have the length change results from observations in-situ or remote sensing (Thomas Schellenberger has the machine learning products)
        Frontal ablation rate [m/a] based on viscoplastic assumptions

    Explanation of sign
    -------
    fa_viscoplastic: negative, mass loss
    dLdt: length change rate, positive if advance; negative if retreat
    terminus mass balance: negative if mass loss; positive if mass gain
    """
    # ---------------------------------------------------------------------------
    # class NegativeValueError(Exception):
    #     pass
    # ---------------------------------------------------------------------------
    ## Global constants
    G = 9.8  # acceleration due to gravity in m/s^2
    RHO_ICE = 900.0  # ice density kg/m^3
    RHO_SEA = 1020.0  # seawater density kg/m^3 
    RHO_WATER = 1000.0  # freshwater density kg/m^3
    #water_level = mb_model.water_level
    print("water_level in the fa_sermeq_speed_law_inv at the start is (m) :",water_level)
    print("variable_yield is ", variable_yield)
    if variable_yield is None and not variable_yield:
        variable_yield = None
#    if water_level is None:
#        water_level = 0
#    rho = cfg.PARAMS['ice_density']
#    rho_o = cfg.PARAMS['ocean_density']
    # ---------------------------------------------------------------------------
    # the yield strength
    def tau_y(tau0=tau0, variable_yield=None, bed_elev=None, thick=None, mu=0.01,water_depth =None):
        """
        Functional form of yield strength.
        Can do constant or Mohr-Coulomb yield strength.  Ideally, the glacier's yield type
        ('constant' or 'variable') would be saved in a model instance.

        Parameters
        ----------
        tau0: float, optional
            Initial guess for yield strength [Pa]. Default is 150 kPa.
        yield_type: str, optional
            'constant' or 'variable' (Mohr-Coulomb) yielding. Default is constant.
        bed_elev: float, optional
            Bed elevation, dimensional [m]. The default is None.
        thick: float, optional
            Ice thickness, dimensional [m]. The default is None.
        mu: float, optional
            Mohr-Coulomb cohesion, a coefficient between 0 and 1. Default is 0.01.

        Returns
        -------
        tau_y: float
            The yield strength for these conditions.
        """
        tau1=tau0*1e5
        if variable_yield is not None:
            if water_depth is None:
                try:
                    # if bed_elev < 0:
                    #     D = -1 * bed_elev  # Water depth D the nondim bed topography value when Z<0
                    # else:
                    #     D = 0
                    D = utils.clip_min(0,water_level - bed_elev)
                except:
                    print('You must set a bed elevation and ice thickness to use variable yield strength. Using constant yeild instead')
            else:
                D = water_depth
            N = RHO_ICE * G * thick - RHO_SEA * G * D  # Normal stress at bed
            #convert to Pa
            
            ty = tau1 + mu * N
        else:  # assume constant if not set
            ty = tau1
        print("the yield strength now is PA:",ty)
        return ty


    # ---------------------------------------------------------------------------
    # calculate the yield ice thickness

    def balance_thickness(yield_strength, bed_elev = None,water_depth = None):
        """
        Ice thickness such that the stress matches the yield strength.

        Parameters
        ----------
        yield_strength: float
            The yield strength near the terminus.
            If yield type is constant, this will of course be the same everywhere.  If yield type is
            variable (Mohr-Coulomb), the yield strength at the terminus could differ from elsewhere.
        bed_elev: float
            Elevation of glacier bed at the terminus

        Returns
        -------
        Hy: float
            The ice thickness for stress balance at the terminus. [units m]
        """
        if water_depth is None:
            # if bed_elev < 0:
            #     D = -1 * bed_elev
            # else:
            #     D = 0
            D = utils.clip_min(0,water_level - bed_elev)
        else:
            D = water_depth
        return (2 * yield_strength / (RHO_ICE * G)) + np.sqrt(
            (RHO_SEA * (D ** 2) / RHO_ICE) + ((2 * yield_strength / (RHO_ICE * G)) ** 2))
        # TODO: Check on exponent on last term.  In Ultee & Bassis 2016, this is squared, but in Ultee & Bassis 2020 supplement, it isn't.


    # ---------------------------------------------------------------------------
    # calculate frontal ablation based on the ice thickness, speed at the terminus
    fls = gdir.read_pickle('inversion_flowlines',filesuffix=filesuffix)
    glacier_area = fls[-1].widths_m * fls[-1].dx_meter
    flowline=fls[-1]
    #print("dir of flowline in inversion_flowlines is:",dir(flowline))
    cls = gdir.read_pickle('inversion_output',filesuffix=filesuffix)
    cl = cls[-1]
    #print("dir of the cl in inversion_output:",dir(cl))
    surface_m = flowline.surface_h
    print("surface_h is (m a.s.l.):",surface_m)
#    f_b = utils.clip_min(1e-3,cl['hgt'][-1] - water_level)
#    out_thick[-1] = (((RHO_SEA / RHO_ICE) * min_rel_h * f_b) / 
#                     ((RHO_SEA / RHO_ICE) * min_rel_h - min_rel_h + 1))
    bed_m = cl['hgt'] - cl['thick']
    #bed_m=surface_m-cl['thick']
    print("bed_m is (m a.s.l.):",bed_m)
    #bed_m = flowline.bed_h
    width_m = flowline.widths_m
    print('width_m (m):',width_m)
    #velocity_m = model.u_stag[-1]*cfg.SEC_IN_YEAR

    section = cl['volume'] / cl['dx']
    print("section is :",section)
    # this flux is in m3 per second
    flux = cl['flux']
    print("flux is (m3 per second) :",flux)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        velocity = flux / section
    velocity *= cfg.SEC_IN_YEAR
    print("velocity is (m a-1):",velocity)
    velocity_m = velocity
    #x_m = flowline.dis_on_line*flowline.map_dx/1000
    print("flowline.dis_on_line is :",flowline.dis_on_line)
    print("flowline.map_dx is :",flowline.map_dx)
    x_m = flowline.dis_on_line*flowline.map_dx
    print("x_m is (m, distance along flowline, 0 is the glacier head):",x_m)
    # gdir : py:class:`oggm.GlacierDirectory`
    #     the glacier directory to process
    # fls = model.gdir.read_pickle('model_flowlines')
    # mbmod_fl = massbalance.MultipleFlowlineMassBalance(model.gdir, fls=fls, use_inversion_flowlines=True,
    #                                                    mb_model_class=MonthlyTIModel)
    #mb_annual=mb_model.get_annual_mb(heights=surface_m, fl_id=-1, year=mb_years, fls=fls)
    
    #mb_annual=flowline.apparent_mb
    # call the real climatic mass balance from PyGEMMassBalance
    #mb_annual=flowline.apparent_mb

    #  should call the monthly function
    #mb_annual=mb_model.get_monthly_mb(heights=surface_m, fl_id=-1, year=model.yr, fls=fls)
    # Get the mean mass balance for the study period along the flowline
    # Set model parameters
        # ----- Invert ice thickness and run simulation ------
    
    if (fls is not None) and (glacier_area.sum() > 0): 
        try:    
            
            #mean_mb_model = ConstantMassBalance(gdir,mb_model_class = mb_model.__class__,modelprms = modelprms,
            #                                    glacier_rgi_table = glacier_rgi_table,
            #                                    hindcast = hindcast,
            #                                    debug = debug,
            #                                    debug_refreeze = debug_refreeze,
            #                                    fls = fls, option_areaconstant = option_areaconstant,
            #                                    inversion_filter = inversion_filter, y0=2010, halfsize=10)


            # check that this gives you 2000, 2001, ..., 2020
            #print("mean mb model for the period is:",mean_mb_model.years)
            
            #TODO add the fls and fl_id in the input argument of function mean_mb_model.get_annual_mb(heights=surface_m)
            #fls = gdir.read_pickle('inversion_flowlines')
            # print("######################### before call the get_annual_mb function #########################")
            # print("fls is : ",fls)
            # fl_id = 0
            # print("fl_id is :",fl_id)
            # print("######################### before call the get_annual_mb function #########################")
            # mean_mb_annual=surface_m * 0.
            # for y in range(len(mb_years)):
            #     mean_mb_annual+=mb_model.get_annual_mb(fls = fls, fl_id = -1, heights=surface_m, year=y)
            #     print('y in fa_sermeq_speed_law_inv is :',y)
            # mean_mb_annual=mean_mb_annual/float(len(mb_years))
            # print("mean_mb_annual is (m ice per second):",mean_mb_annual)

            # Terminus_mb = mean_mb_annual*cfg.SEC_IN_YEAR
            #Note: In the function, fa_sermeq_speed_law_inv, change the mb_annual = flowline.apparent_mb,
            # to make consistent with the apparent flux of the flowline, in the bed inversion process
            mb_annual=flowline.apparent_mb # the unit is mm w.e. a-1 or kg m-2 a-1 of ice
            
            # convert the unit from kg m-2 a-1 of ice to m a-1 of ice, 
            Terminus_mb = mb_annual/RHO_ICE
            print("Terminus_mb is (m a-1  / m ice per year):",Terminus_mb)
            # slice up to index+1 to include the last nonzero value
            # profile: NDarray
            #     The current profile (x, surface, bed,width) as calculated by the base model
            #     Unlike core SERMeQ, these should be DIMENSIONAL [m].
            print("The last above wl is (In fa_sermeq_speed_law_inv) :",last_above_wl)
            print("WATER LEVEL IS :",water_level)
            #pdb.set_trace()
            profile=(x_m[:last_above_wl+1],
                        surface_m[:last_above_wl+1],
                        bed_m[:last_above_wl+1],width_m[:last_above_wl+1])
            # model_velocity: array
            #     Velocity along the flowline [m/a] as calculated by the base model
            #     Should have values for the points nearest the terminus...otherwise
            #     doesn't matter if this is the same shape as the profile array.
            #     TODO: Check with the remote sensing products, or at least to validate the model products
            model_velocity=velocity_m[:last_above_wl+1]
            # remove lowest cells if needed
            print("trim_profile is :",trim_profile)
            last_index = -1 * (trim_profile + 1)
            print("the last index is :", last_index)
            ## TODO: Check the flowline model, the decrease the distance between two adjacent points along the flowline, and then calculate the averaged gradient for dhdx,dhydx,dudx
            ##
            if isinstance(Terminus_mb, (int, float)):
                terminus_mb = Terminus_mb
            elif isinstance(Terminus_mb, (list, np.ndarray)):
                terminus_mb = Terminus_mb[last_index]
            else:
                print("please input the correct mass balance datatype")
            #
            if isinstance(model_velocity, (int, float)):
                model_velocity = v_scaling * model_velocity
            elif isinstance(model_velocity, list):
                model_velocity = v_scaling * np.array(model_velocity)
            elif isinstance(model_velocity, np.ndarray):
                model_velocity = v_scaling * model_velocity
            else:
                print("please input the correct velocity datatype")
            ## Ice thickness and yield thickness nearest the terminus
            se_terminus = profile[1][last_index]
            bed_terminus = profile[2][last_index]
            h_terminus = se_terminus - bed_terminus
            #h_terminus_T = thick_T # the thickness revised based on water depth and free_board
            width_terminus = profile[3][last_index]
            tau_y_terminus = tau_y(tau0=tau0, bed_elev=bed_terminus, thick=h_terminus, variable_yield=variable_yield)
            print('tau_y_terminus (Pa) is :',tau_y_terminus)
            Hy_terminus = balance_thickness(yield_strength=tau_y_terminus, bed_elev=bed_terminus)
            print('Hy_terminus:',Hy_terminus)
            if isinstance(model_velocity, (int, float)):
                U_terminus = model_velocity
                U_adj = model_velocity
            else:
                U_terminus = model_velocity[last_index]  ## velocity, assuming last point is terminus
                U_adj = model_velocity[last_index - 1]
            ## Ice thickness and yield thickness at adjacent point
            se_adj = profile[1][last_index - 1]
            bed_adj = profile[2][last_index - 1]
            H_adj = se_adj - bed_adj
            #H_adj_T= h_terminus_T*H_adj/h_terminus # derived based on h_terminus_T
            #water_depth_adj = water_depth+bed_terminus-bed_adj
            
            tau_y_adj = tau_y(tau0=tau0, bed_elev=bed_adj, thick=H_adj, variable_yield=variable_yield)
            Hy_adj = balance_thickness(yield_strength=tau_y_adj, bed_elev=bed_adj)
            print('Hy_adj:',Hy_adj)
            # Gradients
            dx_term = profile[0][last_index] - profile[0][last_index - 1]  ## check grid spacing close to terminus
            dHdx = (h_terminus - H_adj) / dx_term
            dHydx = (Hy_terminus - Hy_adj) / dx_term
            if np.isnan(U_terminus) or np.isnan(U_adj):
                dUdx = np.nan  ## velocity gradient
                ## Group the terms
                dLdt_numerator = np.nan
                dLdt_denominator = np.nan  ## TODO: compute dHydx
                dLdt_viscoplastic = np.nan
                # fa_viscoplastic = dLdt_viscoplastic -U_terminus  ## frontal ablation rate
                fa_viscoplastic = np.nan  ## frontal ablation rate
            else:
                # Gradients
                # dx_term = profile[0][last_index] - profile[0][last_index - 1]  ## check grid spacing close to terminus
                # dHdx = (h_terminus - H_adj) / dx_term
                # dHydx = (Hy_terminus - Hy_adj) / dx_term
                                # consider the dudx refers to the strain rate, here should be make sure dUdx is >=0
                dUdx = (U_terminus - U_adj) / dx_term  ## velocity gradient
                ## Group the terms
                dLdt_numerator = terminus_mb - (h_terminus* dUdx) - (U_terminus * dHdx)
                dLdt_denominator = dHydx - dHdx  ## TODO: compute dHydx
                dLdt_viscoplastic = dLdt_numerator / dLdt_denominator
                print('dLdt_numerator',dLdt_numerator)
                print('dLdt_denominator',dLdt_denominator)
                # fa_viscoplastic = dLdt_viscoplastic -U_terminus  ## frontal ablation rate
                
                # try:
                U_calving = U_terminus - dLdt_viscoplastic  ## frontal ablation rate
                fa_viscoplastic=U_calving

                # if U_calving<0:
                #     print("The glacier is advancing, and the advancing rate is larger than ice flow speed at the terminus, please check ")
                #     if U_calving>0 or U_calving==0:
                #         fa_viscoplastic=U_calving
                #     else:
                #         fa_viscoplastic=U_calving
                #         # fa_viscoplastic=np.nan
                #         raise NegativeValueError("Something is wrong, right now the calving in negative, which should be positive or zero")
                # except NegativeValueError as e:
                #     print ("The glacier is advancing, and the advancing rate is larger than ice flow speed at the terminus, please check ")
                    


            SQFA = {'se_terminus': se_terminus,
                    'bed_terminus': bed_terminus,
                    'Thickness_termi': h_terminus,
                    'Width_termi':  width_terminus,
                    'Hy_thickness': Hy_terminus,
                    'Velocity_termi': U_terminus,
                    'Terminus_mb': terminus_mb,
                    'dLdt': dLdt_viscoplastic,
                    'Sermeq_fa': fa_viscoplastic}
            if verbose:
                print('For inspection on debugging - all should be DIMENSIONAL (m/a):')
                #         print('profile_length={}'.format(profile_length))
                print('last_index={}'.format(last_index))
                print('se_terminus={}'.format(se_terminus))
                print('bed_terminus={}'.format(bed_terminus))
                print('se_adj={}'.format(se_adj))
                print('bed_adj={}'.format(bed_adj))
                print('Thicknesses: Hterm {}, Hadj {}'.format(h_terminus, H_adj))
                print('Hy_terminus={}'.format(Hy_terminus))
                print('Hy_adj={}'.format(Hy_adj))
                print('U_terminus={}'.format(U_terminus))
                print('U_adj={}'.format(U_adj))
                print('dUdx={}'.format(dUdx))
                print('dx_term={}'.format(dx_term))
                print('Checking dLdt: terminus_mb = {}. \n H dUdx = {}. \n U dHdx = {}.'.format(terminus_mb, dUdx * Hy_terminus,
                                                                                                U_terminus * dHdx))
                print('Denom: dHydx = {} \n dHdx = {}'.format(dHydx, dHdx))
                print('Viscoplastic dLdt={}'.format(dLdt_viscoplastic))
                print('Terminus surface mass balance ma= {}'.format(terminus_mb))
                print('Sermeq frontal ablation ma={}'.format(fa_viscoplastic))
            else:
                pass
            return SQFA
        except:
            print("Something wrong in the try part in fa_sermeq_speed_law_inv,in inversion_RT_New.py")
            print(traceback.format_exc())



@entity_task(log, writes=['inversion_output'])
def mass_conservation_inversion(gdir, glen_a=None, fs=None, write=True,
                                filesuffix='', water_level=None,
                                t_lambda=None, min_rel_h=1, 
                                fixed_water_depth=None):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    glen_a : float
        glen's creep parameter A. Defaults to cfg.PARAMS.
    fs : float
        sliding parameter. Defaults to cfg.PARAMS.
    write: bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    filesuffix : str
        add a suffix to the output file
    water_level : float
        to compute volume below water level - adds an entry to the output dict
    t_lambda : float
        defining the angle of the trapezoid walls (see documentation). Defaults
        to cfg.PARAMS.
    min_rel_h : float
        inverse of relative height above buoancy at the terminus given by
        ice thickness inversion with calving
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']

    # Check input
    print("==================== mass_conservation_inversion Start ====================")
    print("fs is (the sliding parameter is):",fs)
    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    a3 = fs / fd
    rho = cfg.PARAMS['ice_density']
    rho_o = cfg.PARAMS['ocean_density']

    if water_level is None:
        water_level = 0
    print("water_level in the mass_conservation_inversion is :",water_level)
    # Inversion with shape factors?
    sf_func = None
    use_sf = cfg.PARAMS.get('use_shape_factor_for_inversion', None)
    if use_sf == 'Adhikari' or use_sf == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif use_sf == 'Huss':
        sf_func = utils.shape_factor_huss

    print("shape factor is (sf_func):",sf_func)
    
    # Clip the slope, in rad
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    print("min slope is:",min_slope)

    out_volume = 0.
    do_calving = cfg.PARAMS['use_kcalving_for_inversion'] and gdir.is_tidewater
    cls = gdir.read_pickle('inversion_input',filesuffix=filesuffix)
    # for cl in cls:
    #     # Clip slope to avoid negative and small slopes
    #     slope = cl['slope_angle']
    #     slope = utils.clip_array(slope, min_slope, np.pi/2.)

    #     # Glacier width
    #     w = cl['width']
    for n_cl, cl in enumerate(cls):
        print("the ", n_cl,"flowline")
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = utils.clip_array(slope, min_slope, np.pi/2.)
        sf = np.ones(slope.shape)  #  Default shape factor is 1
        h_tol = 0.1
        max_iter = 200
        # Glacier width
        w = cl['width']
        k = 0
        h_diff = np.ones(slope.shape)
        a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)
        out_thick = _compute_thick(a0s, a3, cl['flux_a0'], sf, _inv_function)
        #print("out_thick before mass_conservation_inversion is :",out_thick)
        # Iteratively seeking glacier state including water-depth dependent
        # processes
        while k < max_iter and np.any(h_diff > h_tol):
            print("=========== the ",k, "iteration in mass_conservation_inversion ===========")
            h_diff = out_thick
            
            if do_calving and n_cl==(len(cls)-1) and min_rel_h > 1:
                print("***************** min_rel_h is >1 *****************")
                f_b = utils.clip_min(1e-3,cl['hgt'][-1] - water_level)
                out_thick[-1] = (((rho_o / rho) * min_rel_h * f_b) / 
                                 ((rho_o / rho) * min_rel_h - min_rel_h + 1))
                bed_h = cl['hgt'] - out_thick
                water_depth = utils.clip_min(0,-bed_h+water_level)
                if fixed_water_depth:
                    out_thick[-1] = ((min_rel_h * (rho_o / rho) * 
                                      fixed_water_depth) / (min_rel_h - 1))
                    water_depth[-1] = fixed_water_depth
                rel_h = out_thick / (out_thick - (rho_o / rho) * water_depth)
                wrong_rel_h = ((rel_h < 1) | ~np.isfinite(rel_h))
                rel_h[wrong_rel_h & (water_depth > 0)] = min_rel_h
                rel_h[wrong_rel_h & (water_depth == 0)] = 1
                rel_h[-1] = min_rel_h
                a_pull = a0s * 0
                length = len(cl['width']) * cl['dx']
                stretch_dist_p = cfg.PARAMS.get('stretch_dist', 8e3)
                stretch_dist = utils.clip_max(stretch_dist_p,length)
                stretch_dist = utils.clip_max(stretch_dist, length)
                n_stretch = np.rint(stretch_dist / cl['dx']).astype(int)
                pull_stress = utils.clip_min(0, 0.5 * cfg.G * (rho *
                                             out_thick[-1]**2 - rho_o *
                                             water_depth[-1]**2))

                # Define stretch factor and add to driving stress
                stretch_factor = np.zeros(n_stretch)
                for j in range(n_stretch):
                    stretch_factor[j] = 2*(j+1)/(n_stretch+1)
                if cl['dx'] > stretch_dist:
                    stretch_factor = stretch_dist / cl['dx']
                    n_stretch = 1

                a_pull[-(n_stretch):] = (stretch_factor * (pull_stress /
                                         stretch_dist))
                a_factor = (a_pull / (rho*cfg.G*slope*out_thick)) + 1
                a_factor = np.nan_to_num(a_factor, nan=1, posinf=1, neginf=1)
            else:
                print("***************** min_rel_h is <=1 *****************")
                rel_h = out_thick * 0 + 1
                a_factor = out_thick * 0 + 1
            a3s = fs / fd * rel_h
            a0s = - cl['flux_a0'] / ((rho*cfg.G*slope*a_factor)**3 * fd)
            out_thick = _compute_thick(a0s, a3s, cl['flux_a0'], sf,
                                          _inv_function)
            out_thick = utils.clip_min(out_thick, 1.5e-3) # To prevent errors

            if sf_func is not None:
                print("sf_func is not none")
                # Start iteration for shape factor with first guess of 1
                i = 0
                sf_diff = np.ones(slope.shape)

                # Some hard-coded factors here
                sf_tol = 1e-2
                max_sf_iter = 20

                while i < max_sf_iter and np.any(sf_diff > sf_tol):
                    print("the",i,"iteration under sf_func is not none")
                    out_thick = _compute_thick(a0s, a3, cl['flux_a0'], sf,
                                               _inv_function)
                    sf_diff[:] = sf[:]
                    sf = sf_func(w, out_thick, cl['is_rectangular'])
                    sf_diff = sf_diff - sf
                    i += 1

                log.info('Shape factor {:s} used, took {:d} iterations for '
                         'convergence.'.format(use_sf, i))

                # TODO: possible shape factor optimisations
                # thick update could be used as iteration end criterion instead
                # we iterate for all grid points, even if some already converged


    #     a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)

    #     out_thick = _compute_thick(a0s, a3, cl['flux_a0'], _inv_function)

            # volume
            is_rect = cl['is_rectangular']
            fac = np.where(is_rect, 1, 2./3.)
            volume = fac * out_thick * w * cl['dx']

            # Now recompute thickness where parabola is too flat
            print("========== Start to recompute thickness where parabola is too flat ========== ")
            is_trap = cl['is_trapezoid']
            if cl['invert_with_trapezoid']:
                min_shape = cfg.PARAMS['mixed_min_shape']
                bed_shape = 4 * out_thick / w ** 2
                is_trap = ((bed_shape < min_shape) & ~ cl['is_rectangular'] &
                           (cl['flux'] > 0)) | is_trap
                for i in np.where(is_trap)[0]:
                    try:
                        out_thick[i] = sia_thickness_via_optim(slope[i], w[i],
                                                               cl['flux'][i],
                                                               rel_h[i],
                                                               a_factor[i],
                                                               shape='trapezoid',
                                                               t_lambda=t_lambda,
                                                               glen_a=glen_a,
                                                               fs=fs)
                        sect = (2*w[i] - t_lambda * out_thick[i]) / 2 * out_thick[i]
                        volume[i] = sect * cl['dx']
                    except ValueError:
                        # no solution error - we do with rect
                        out_thick[i] = sia_thickness_via_optim(slope[i], w[i],
                                                               cl['flux'][i],
                                                               rel_h[i],
                                                               a_factor[i],
                                                               shape='rectangular',
                                                               glen_a=glen_a,
                                                               fs=fs)
                        is_rect[i] = True
                        is_trap[i] = False
                        volume[i] = out_thick[i] * w[i] * cl['dx']
            h_diff = h_diff - out_thick
            k += 1
            print("k in the iterations of mass_conservation_inversion now is:",k)
        print("--------------------------------------------------------")
        #print("h_diff is :",h_diff)
        print("h_tol is :",h_tol)
        print("np.any(h_diff > h_tol) is :",np.any(h_diff > h_tol))
        print("--------------------------------------------------------")
        # Sanity check
        if np.any(out_thick <= -1e-2):
            log.warning(f"{gdir.rgi_id} Found negative thickness: "
                        f"n={(out_thick <= -1e-2).sum()}, "
                        f"v={np.min(out_thick)}.")

        out_thick = utils.clip_min(out_thick, 0)
        #print("the out_thick after mass_conservation_inversion is :",out_thick)
        if write:
            cl['is_trapezoid'] = is_trap
            cl['is_rectangular'] = is_rect
            cl['thick'] = out_thick
            cl['volume'] = volume

            # volume below sl
            try:
                bed_h = cl['hgt'] - out_thick
                bed_shape = 4 * out_thick / w ** 2
                if np.any(bed_h < 0):
                    cl['volume_bsl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick, w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'], 0)
                if water_level is not None and np.any(bed_h < water_level):
                    cl['volume_bwl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick, w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'], water_level)
            except KeyError:
                # cl['hgt'] is not available on old prepro dirs
                pass

        out_volume += np.sum(volume)

    if write:
        gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)
        gdir.add_to_diagnostics('inversion_glen_a', glen_a,filesuffix=filesuffix)
        gdir.add_to_diagnostics('inversion_fs', fs,filesuffix=filesuffix)

    #print("==================== mass_conservation_inversion is successful ====================")

    return out_volume


@entity_task(log, writes=['inversion_output'])
def filter_inversion_output(gdir, n_smoothing=5, min_ice_thick=1.,
                            max_depression=5.):
    """Filters the last few grid points after the physically-based inversion.

    For various reasons (but mostly: the equilibrium assumption), the last few
    grid points on a glacier flowline are often noisy and create unphysical
    depressions. Here we try to correct for that. It is not volume conserving,
    but area conserving. If a parabolic shape factor is getting smaller than
    the minimum defined one (cfg.PARAMS['mixed_min_shape']) the grid point is
    changed to a trapezoid, similar to what is done during the actual
    physically-based inversion.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    n_smoothing : int
        number of grid points which should be smoothed. Default is 5
    min_ice_thick : float
        the minimum ice thickness after the smoothing. Default is 1 m
    max_depression : float
        the limit allowed bed depression without smoothing. Default is 5 m
    """

    if gdir.is_tidewater:
        # No need for filter in tidewater case
        cls = gdir.read_pickle('inversion_output')
        init_vol = np.sum([np.sum(cl['volume']) for cl in cls])
        return init_vol

    if not gdir.has_file('downstream_line'):
        raise InvalidWorkflowError('filter_inversion_output now needs a '
                                   'previous call to the '
                                   'compute_dowstream_line and '
                                   'compute_downstream_bedshape tasks')

    dic_ds = gdir.read_pickle('downstream_line')
    cls = gdir.read_pickle('inversion_output')
    cl = cls[-1]

    # check that their are enough grid points for smoothing
    nr_grid_points = len(cl['thick'])
    if nr_grid_points <= n_smoothing:
        if nr_grid_points >= 3:
            n_smoothing = nr_grid_points - 1
        else:
            log.warning(f'({gdir.rgi_id}) filter_inversion_output: flowline '
                        f'has not enough grid points for applying the filter '
                        f'(only {nr_grid_points} grid points)!')
            # Return volume for convenience
            return np.sum([np.sum(cl['volume']) for cl in cls])

    # convert to negative number for indexing
    n_smoothing = -abs(n_smoothing)

    cl_sfc_h = cl['hgt'][n_smoothing:]
    cl_thick = cl['thick'][n_smoothing:]
    cl_width = cl['width'][n_smoothing:]
    cl_is_trap = cl['is_trapezoid'][n_smoothing:]
    cl_is_rect = cl['is_rectangular'][n_smoothing:]
    cl_bed_h = cl_sfc_h - cl_thick
    try:
        downstream_sfc_h = dic_ds['surface_h'][:5]
    except KeyError:
        raise InvalidWorkflowError('Please run compute_downstream_line and '
                                   'compute_downstream_bedshape for the '
                                   'filter.')

    # we smooth if the depression is larger than max_depression
    if downstream_sfc_h[0] - cl_bed_h[-1] > max_depression:
        # force the last grid point height to continue the downstream slope
        down_slope_avg = np.average(np.abs(np.diff(downstream_sfc_h)))
        new_last_bed_h = downstream_sfc_h[0] + down_slope_avg

        # now smoothly add the change of the bed height over all smoothing grid
        # points
        all_bed_h_changes = np.linspace(0, new_last_bed_h - cl_bed_h[-1],
                                        -n_smoothing)
        cl_bed_h = cl_bed_h + all_bed_h_changes

    # define new thick and clip, maximum value needed to avoid geometrical
    # inconsistencies with trapezoidal bed shape
    new_thick = cl_sfc_h - cl_bed_h
    # max
    max_h = np.where(cl_is_trap,
                     # -1. to get w0 > 0 at the end and not w0 = 0
                     cl_width / cfg.PARAMS['trapezoid_lambdas'] - 1.,
                     1e4)
    new_thick = np.where(new_thick > max_h, max_h, new_thick)
    # min
    new_thick = np.where(new_thick < min_ice_thick, min_ice_thick, new_thick)

    # new trap = (is trap or shape factor small) and not rectangular, we
    # conserve all bed shapes
    # if new bed shape is smaller than defined minimum it is converted to a
    # trapezoidal (as it is done during inversion)
    new_bed_shape = 4 * new_thick / cl_width ** 2
    new_is_trapezoid = ((cl_is_trap |
                         (new_bed_shape < cfg.PARAMS['mixed_min_shape'])) &
                        ~cl_is_rect)

    # and calculate new volumes depending on shape
    new_volume = np.where(
        new_is_trapezoid,
        cl_width * new_thick - cfg.PARAMS['trapezoid_lambdas'] / 2 *
        new_thick ** 2,
        np.where(cl_is_rect, 1, 2 / 3) * cl_width * new_thick) * cl['dx']

    # define new values
    cl['thick'][n_smoothing:] = new_thick
    cl['is_trapezoid'][n_smoothing:] = new_is_trapezoid
    cl['volume'][n_smoothing:] = new_volume

    gdir.write_pickle(cls, 'inversion_output')

    # Return volume for convenience
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log)
def get_inversion_volume(gdir):
    """Small utility task to get to the volume od all glaciers."""
    cls = gdir.read_pickle('inversion_output')
    return np.sum([np.sum(cl['volume']) for cl in cls])


@entity_task(log, writes=['inversion_output'])
def compute_velocities(*args, **kwargs):
    """Deprecated. Use compute_inversion_velocities instead."""
    warnings.warn("`compute_velocities` has been renamed to "
                  "`compute_inversion_velocities`. Prefer to use the new"
                  "name from now on.")
    return compute_inversion_velocities(*args, **kwargs)


@entity_task(log, writes=['inversion_output'])
def compute_inversion_velocities(gdir, glen_a=None, fs=None, filesuffix='',
                                 with_sliding=False):
    """Surface velocities along the flowlines from inverted ice thickness.

    Computed following the methods described in Cuffey and Paterson (2010)
    Eq. 8.35, pp 310:

        u_s = u_basal + (2A/n+1)* tau^n * H

    In the case of no sliding (or if with_sliding=False, which is a
    justifiable simplification given uncertainties on basal sliding):

        u_z/u_s = [n+1]/[n+2] (= 0.8 if n = 3).

    The output is written in 'inversion_output.pkl' in m yr-1

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    with_sliding : bool
        if set to True, we will compute velocities the sliding component.
        the default is to check if sliding was used for the inversion
        (fs != 0)
    glen_a : float
        Glen's deformation parameter A. Defaults to PARAMS['inversion_glen_a']
    fs : float
        sliding paramter, defaults to PARAMS['inversion_fs']
    filesuffix : str
        add a suffix to the output file (optional)
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']

    rho = cfg.PARAMS['ice_density']
    glen_n = cfg.PARAMS['glen_n']

    if with_sliding is None:
        with_sliding = fs != 0

    # Getting the data for the main flowline
    cls = gdir.read_pickle('inversion_output')

    for cl in cls:
        # vol in m3 and dx in m
        section = cl['volume'] / cl['dx']

        # this flux is in m3 per second
        flux = cl['flux']
        angle = cl['slope_angle']
        thick = cl['thick']

        if fs > 0 and with_sliding:
            tau = rho * cfg.G * angle * thick

            with warnings.catch_warnings():
                # This can trigger a divide by zero Warning
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                u_basal = fs * tau ** glen_n / thick
            # TODO: add water-depth dependent formulation for water-terminating
            # glaciers
            u_basal[~np.isfinite(u_basal)] = 0

            u_deformation = (2 * glen_a / (glen_n + 1)) * (tau**glen_n) * thick

            u_basal *= cfg.SEC_IN_YEAR
            u_deformation *= cfg.SEC_IN_YEAR
            u_surface = u_basal + u_deformation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                velocity = flux / section
            velocity *= cfg.SEC_IN_YEAR
        else:
            # velocity in cross section
            fac = (glen_n + 1) / (glen_n + 2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                velocity = flux / section
            velocity *= cfg.SEC_IN_YEAR
            u_surface = velocity / fac
            u_basal = velocity * np.NaN
            u_deformation = velocity * np.NaN

        # output
        cl['u_integrated'] = velocity
        cl['u_surface'] = u_surface
        cl['u_basal'] = u_basal
        cl['u_deformation'] = u_deformation

    gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_per_altitude(gdir, add_slope=True,
                                      topo_variable='topo_smoothed',
                                      smooth_radius=None,
                                      dis_from_border_exp=0.25,
                                      varname_suffix=''):
    """Compute a thickness map by redistributing mass along altitudinal bands.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX or
    for visualizations.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_slope : bool
        whether a corrective slope factor should be used or not
    topo_variable : str
        the topography to read from `gridded_data.nc` (could be smoothed, or
        smoothed differently).
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    dis_from_border_exp : float
        the exponent of the distance from border mask
    varname_suffix : str
        add a suffix to the variable written in the file (for experiments)
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with utils.ncDataset(grids_file) as nc:
        has_masks = 'glacier_ext_erosion' in nc.variables
    if not has_masks:
        from oggm.core.gis import gridded_attributes
        gridded_attributes(gdir)

    with utils.ncDataset(grids_file) as nc:
        topo = nc.variables[topo_variable][:]
        glacier_mask = nc.variables['glacier_mask'][:]
        dis_from_border = nc.variables['dis_from_border'][:]
        if add_slope:
            slope_factor = nc.variables['slope_factor'][:]
        else:
            slope_factor = 1.

    # Along the lines
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    hs, ts, vs, xs, ys = [], [], [], [], []
    for cl, fl in zip(cls, fls):
        hs = np.append(hs, fl.surface_h)
        ts = np.append(ts, cl['thick'])
        vs = np.append(vs, cl['volume'])
        try:
            x, y = fl.line.xy
        except AttributeError:
            # Squeezed flowlines, dummy coords
            x = fl.surface_h * 0 - 1
            y = fl.surface_h * 0 - 1
        xs = np.append(xs, x)
        ys = np.append(ys, y)

    init_vol = np.sum(vs)

    # Assign a first order thickness to the points
    # very inefficient inverse distance stuff
    thick = glacier_mask * 0.
    yglac, xglac = np.nonzero(glacier_mask == 1)
    for y, x in zip(yglac, xglac):
        phgt = topo[y, x]
        # take the ones in a 100m range
        starth = 100.
        while True:
            starth += 10
            pok = np.nonzero(np.abs(phgt - hs) <= starth)[0]
            if len(pok) != 0:
                break
        sqr = np.sqrt((xs[pok]-x)**2 + (ys[pok]-y)**2)
        pzero = np.where(sqr == 0)
        if len(pzero[0]) == 0:
            thick[y, x] = np.average(ts[pok], weights=1 / sqr)
        elif len(pzero[0]) == 1:
            thick[y, x] = ts[pzero[0][0]]
        else:
            raise RuntimeError('We should not be there')

    # Distance from border (normalized)
    dis_from_border = dis_from_border**dis_from_border_exp
    dis_from_border /= np.mean(dis_from_border[glacier_mask == 1])
    thick *= dis_from_border

    # Slope
    thick *= slope_factor

    # Smooth
    dx = gdir.grid.dx
    if smooth_radius != 0:
        if smooth_radius is None:
            smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, int(smooth_radius))
        thick = np.where(glacier_mask, thick, 0.)

    # Re-mask
    utils.clip_min(thick, 0, out=thick)
    thick[glacier_mask == 0] = np.NaN
    assert np.all(np.isfinite(thick[glacier_mask == 1]))

    # Conserve volume
    tmp_vol = np.nansum(thick * dx**2)
    thick *= init_vol / tmp_vol

    # write
    with utils.ncDataset(grids_file, 'a') as nc:
        vn = 'distributed_thickness' + varname_suffix
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = '-'
        v.long_name = 'Distributed ice thickness'
        v[:] = thick

    return thick


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_interp(gdir, add_slope=True, smooth_radius=None,
                                varname_suffix=''):
    """Compute a thickness map by interpolating between centerlines and border.

    IMPORTANT: this is NOT what has been used for ITMIX. We used
    distribute_thickness_per_altitude for ITMIX and global ITMIX.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_slope : bool
        whether a corrective slope factor should be used or not
    smooth_radius : int
        pixel size of the gaussian smoothing. Default is to use
        cfg.PARAMS['smooth_window'] (i.e. a size in meters). Set to zero to
        suppress smoothing.
    varname_suffix : str
        add a suffix to the variable written in the file (for experiments)
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with utils.ncDataset(grids_file) as nc:
        has_masks = 'ice_divides' in nc.variables
    if not has_masks:
        from oggm.core.gis import gridded_attributes
        gridded_attributes(gdir)

    with utils.ncDataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
        glacier_ext = nc.variables['glacier_ext_erosion'][:]
        ice_divides = nc.variables['ice_divides'][:]
        if add_slope:
            slope_factor = nc.variables['slope_factor'][:]
        else:
            slope_factor = 1.

    # Thickness to interpolate
    thick = glacier_ext * np.NaN
    thick[(glacier_ext-ice_divides) == 1] = 0.
    # TODO: domain border too, for convenience for a start
    thick[0, :] = 0.
    thick[-1, :] = 0.
    thick[:, 0] = 0.
    thick[:, -1] = 0.

    # Along the lines
    cls = gdir.read_pickle('inversion_output')
    fls = gdir.read_pickle('inversion_flowlines')
    vs = []
    for cl, fl in zip(cls, fls):
        vs.extend(cl['volume'])
        x, y = utils.tuple2int(fl.line.xy)
        thick[y, x] = cl['thick']
    init_vol = np.sum(vs)

    # Interpolate
    xx, yy = gdir.grid.ij_coordinates
    pnan = np.nonzero(~ np.isfinite(thick))
    pok = np.nonzero(np.isfinite(thick))
    points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
    inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
    thick[pnan] = griddata(points, np.ravel(thick[pok]), inter, method='cubic')
    utils.clip_min(thick, 0, out=thick)

    # Slope
    thick *= slope_factor

    # Smooth
    dx = gdir.grid.dx
    if smooth_radius != 0:
        if smooth_radius is None:
            smooth_radius = np.rint(cfg.PARAMS['smooth_window'] / dx)
        thick = gaussian_blur(thick, int(smooth_radius))
        thick = np.where(glacier_mask, thick, 0.)

    # Re-mask
    thick[glacier_mask == 0] = np.NaN
    assert np.all(np.isfinite(thick[glacier_mask == 1]))

    # Conserve volume
    tmp_vol = np.nansum(thick * dx**2)
    thick *= init_vol / tmp_vol

    # write
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file, 'a') as nc:
        vn = 'distributed_thickness' + varname_suffix
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = '-'
        v.long_name = 'Distributed ice thickness'
        v[:] = thick

    return thick


def calving_flux_from_depth(gdir, mb_model=None,mb_years=None,k=None, water_level=None, water_depth=None,
                            thick=None, fixed_water_depth=False,calving_law_inv=None,modelprms = None,
                            glacier_rgi_table = None, hindcast = None, debug = None, debug_refreeze = None,
                            option_areaconstant = None, inversion_filter = False,filesuffix=''):
    """Finds a calving flux from the calving front thickness.

    Approach based on Huss and Hock, (2015) and Oerlemans and Nick (2005).
    We take the initial output of the model and surface elevation data
    to calculate the water depth of the calving front.

    Parameters
    ----------
    gdir : GlacierDirectory
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    k : float
        calving constant
    water_level : float
        in case water is not at 0 m a.s.l
    water_depth : float (mandatory)
        the default is to compute the water_depth from ice thickness
        at the terminus and altitude. Set this to force the water depth
        to a certain value
    thick :
        Set this to force the ice thickness to a certain value (for
        sensitivity experiments).
    fixed_water_depth :
        If we have water depth from Bathymetry we fix the water depth
        and forget about the free-board
    calving_law_inv : func
            option to use another calving law in the inversion.
            This is a temporary workaround
            to test other calving laws, and the system might be improved in
            future OGGM versions.
            1. k-calving law
            2. fa_sermq_speed_law_inv
    filesuffix : str
        add a suffix to the output file (optional)
    
    # parameters for the mb_model, used in the function fa_sermeq_speed_law_inv
    ----------
    modelprms : dict
        Model parameters dictionary (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
    glacier_rgi_table : pd.Series
        Table of glacier's RGI information
    option_areaconstant : Boolean
        option to keep glacier area constant (default False allows glacier area to change annually)
    frontalablation_k : float
        frontal ablation parameter
    debug : Boolean
        option to turn on print statements for development or debugging of code
    debug_refreeze : Boolean
        option to turn on print statements for development/debugging of refreezing code
    hindcast : Boolean
        switch to run the model in reverse or not (may be irrelevant after converting to OGGM's setup)    

    Returns
    -------
    A dictionary containing:
    - the calving flux in [km3 yr-1]
    - the frontal width in m
    - the frontal thickness in m
    - the frontal water depth in m
    - the frontal free board in m
    """

    # Defaults
    if k is None:
        k = cfg.PARAMS['inversion_calving_k']
    
    print("The inversion calving k is:",k)
    # Read necessary data
    fl = gdir.read_pickle('inversion_flowlines',filesuffix=filesuffix)[-1]

    # Altitude at the terminus and frontal width
    free_board = utils.clip_min(fl.surface_h[-1], 0) - water_level
    width = fl.widths[-1] * gdir.grid.dx

    # Calving formula
    if thick is None:
        cl = gdir.read_pickle('inversion_output',filesuffix = filesuffix)[-1]
        thick = cl['thick'][-1]
        print("the thick in inversion is",thick)
    if water_depth is None:
        water_depth = thick - free_board
        #print("the water_depth in inversion is",water_depth)
    elif not fixed_water_depth:
        # Correct thickness with prescribed water depth
        # If fixed_water_depth=True then we forget about t_altitude
        thick = water_depth + free_board
    print("the water depth is :",water_depth)
    print("the free board is :",free_board)
    print("the thick in inversion now is",thick)
    if calving_law_inv == fa_sermeq_speed_law_inv :
        print("do the fa_sermeq_speed_law_inv")
        # Fa_Sermeq_speed_law to calculate calving
        # We do calving only if there is some ice above wl
        last_above_wl = np.nonzero((fl.surface_h > water_level) &(thick > 0))[0][-1]
        print("the last above wl is :",last_above_wl)
        bed_h = fl.surface_h - cl['thick']
        print("bed_h is (m a.s.l):",bed_h)
        if bed_h[last_above_wl] > water_level:
            print('The terminus bed is still above the water leverl')
        else:
            print('The terminus bed is already under the water level')        #TODO the k should be the updated k, or the defaulted k
        s_fa = fa_sermeq_speed_law_inv(gdir=gdir, mb_model=mb_model,mb_years=mb_years, last_above_wl=last_above_wl,v_scaling = 1, verbose = True,tau0 = k,
                                    mu = 0.01,trim_profile = 1,modelprms = modelprms,glacier_rgi_table = glacier_rgi_table,hindcast = hindcast,
                                    debug = debug, debug_refreeze = debug_refreeze,option_areaconstant = option_areaconstant,
                                    inversion_filter = inversion_filter,water_depth = water_depth,water_level = water_level)
        

        flux = s_fa ['Sermeq_fa']*s_fa['Thickness_termi']*s_fa['Width_termi']/1e9
        flux_k_calving = k * thick * water_depth * width / 1e9 
    else:
        flux = k * thick * water_depth * width / 1e9  
        flux_k_calving = k * thick * water_depth * width / 1e9 
    print("the flux is (km3 a-1);",flux,"based on the calving law is:",
          calving_law_inv," ; flux in k_clving (km3 a-1) is :",flux_k_calving)
    


    if fixed_water_depth:
        # Recompute free board before returning
        free_board = thick - water_depth

    return {'flux': utils.clip_min(flux, 0),
            'width': width,
            'thick': thick,
            'inversion_calving_k': k,
            'water_depth': water_depth,
            'water_level': water_level,
            'free_board': free_board}


@entity_task(log, writes=['diagnostics'])
def find_inversion_calving_from_any_mb(gdir, mb_model=None, mb_years=None,
                                       water_level=None,
                                       glen_a=None, fs=None,calving_law_inv=None,
                                       modelprms =None,glacier_rgi_table = None,hindcast=None,debug =None,
                                       debug_refreeze = None,option_areaconstant=None, inversion_filter = None,
                                       filesuffix=''):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al. (2019) for details. This task is an update to
    `find_inversion_calving` but acting upon the MB residual (i.e. a shift)
    instead of the model temperature sensitivity.

    Parameters
    ----------
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass balance model to use
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        With this parameter, you can produce more realistic values. The default
        is to infer the water level from PARAMS['free_board_lake_terminating']
        and PARAMS['free_board_marine_terminating']
    glen_a : float, optional
    fs : float, optional
    calving_law_inv : func
            option to use another calving law in the inversion.
            This is a temporary workaround
            to test other calving laws, and the system might be improved in
            future OGGM versions.
            1. fa_sermq_speed_law_inv (default)
            2. None (k-calving law)
    filesuffix : str
        add a suffix to the output file (optional)
    # Parameter for the mb_model (Here is for PyGEMMassBalance), used in the calving_flux_from_depth
    ----------
    modelprms : dict
        Model parameters dictionary (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
    glacier_rgi_table : pd.Series
        Table of glacier's RGI information
    option_areaconstant : Boolean
        option to keep glacier area constant (default False allows glacier area to change annually)
    frontalablation_k : float
        frontal ablation parameter
    debug : Boolean
        option to turn on print statements for development or debugging of code
    debug_refreeze : Boolean
        option to turn on print statements for development/debugging of refreezing code
    hindcast : Boolean
        switch to run the model in reverse or not (may be irrelevant after converting to OGGM's setup)    
    """
    from oggm.core import massbalance

    # Defaults
    # if calving_law_inv is None:
    #     calving_law_inv = cfg.PARAMS['calving_law_inv']
    # elif calving_law_inv == fa_sermeq_speed_law_inv:
    #     calving_law_inv = fa_sermeq_speed_law_inv
    # else :
    #     calving_law_inv = None

    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return
    #print("mb_model: " , mb_model)
    #print("mb_years: " , mb_years)
    diag = gdir.get_diagnostics()
    calving_k = diag.get('optimized_k', cfg.PARAMS['inversion_calving_k'])
    rho = cfg.PARAMS['ice_density']
    rho_o = cfg.PARAMS['ocean_density'] # Ocean density, must be >= ice density

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0
    with utils.DisableLogger():
        try:
            massbalance.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                            mb_years=mb_years,filesuffix=filesuffix)
            print("apparent_mb_from_any_mb running successfully")
        except:
            print("Something is wrong with the function apparent_mb_from_any_mb")
            print(traceback.format_exc())
        prepare_for_inversion(gdir,filesuffix=filesuffix)
        try:
            v_ref = mass_conservation_inversion(gdir, water_level=water_level,
                                            glen_a=glen_a, fs=fs,filesuffix=filesuffix)
            print(" mass_conservation_inversion running successfully")
        except:
            print("Something is wrong with the function mass_conservation_inversion")
            print(traceback.format_exc())
        # try:
        #     workflow.calibrate_inversion_from_consensus(gdir, apply_fs_on_mismatch=True,filter_inversion_output =True,volume_m3_reference=None)
        # except:
        #     print("Something is wrong with the function calibrate_inversion_from_consensus")
        #     print(traceback.format_exc())
        # v_ref = tasks.get_inversion_volume(gdir)    

    # Store for statistics
    gdir.add_to_diagnostics('volume_before_calving', v_ref,filesuffix=filesuffix)
    print("volume before calving is:", v_ref)


    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input',filesuffix = filesuffix)[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # Stupidly enough the slope is clipped in the OGGM inversion, not
    # in inversion prepro - clip here
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    slope = utils.clip_array(slope, min_slope, np.pi / 2.)


    # Check that water level is within given bounds
    print("water level is :",water_level)
    cl = gdir.read_pickle('inversion_output',filesuffix =filesuffix)[-1]
    log.workflow('({}) thickness, freeboard before water and frontal ablation:'
                 ' {}, {}'.format(gdir.rgi_id, cl['thick'][-1], cls['hgt'][-1]))
    thick0 = cl['thick'][-1]
    th = cls['hgt'][-1] # cls['hgt'] is the surface elevation
    if water_level is None:
        # For glaciers that are already relatively thick compared to the 
        # freeboard given by the DEM, it seems useful to start with a lower 
        # water level in order not to underestimate the initial thickness.
        #water_level = -thick0/4 if thick0 > 8*th else 0
        #print("water_level is now:,",water_level)
        # else:
        #     #vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        if th < (1-rho/rho_o)*thick0:
            print ("Warning: The terminus of this glacier is floating")
            water_level = th - (1-rho/rho_o)*thick0
        # elif th > 0.3*thick0:
        #     print("Warning: The freeboard of the terminus of this glacier is more than 0.3*thick")
        #     water_level = th - 0.3*thick0
        else:
            water_level = 0
        if gdir.is_lake_terminating:
            water_level = th - cfg.PARAMS['free_board_lake_terminating']            
            #water_level = utils.clip_scalar(0, th - vmax, th - vmin)
    print("water level now is :",water_level)
    # The functions all have the same shape: they decrease, then increase
    # We seek the absolute minimum first
    # def to_minimize(h):
    #     # fl = calving_flux_from_depth(gdir, water_level=water_level,
    #     #                              water_depth=h)
    #     fl = calving_flux_from_depth(gdir, mb_model=mb_model,mb_years=mb_years,k= calving_k,
    #                                  water_level=water_level,water_depth= h,
    #                                  calving_law_inv = fa_sermeq_speed_law_inv, modelprms = modelprms,
    #                                  glacier_rgi_table = glacier_rgi_table,hindcast = hindcast,
    #                                  debug = debug, debug_refreeze = debug_refreeze,
    #                                  option_areaconstant = option_areaconstant,
    #                                  inversion_filter = inversion_filter)

    #     flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
    #     sia_thick = sia_thickness(slope, width, flux, glen_a=glen_a, fs=fs)
    #     print("flux in to_minimize is (m3 s-1) :",flux)
    #     print("sia_thick in to_minimize is :",sia_thick)
    #     return fl['thick'] - sia_thick #

    def to_minimize(rel_h):
        f_b = th - water_level
        thick = ((rho_o/rho)*rel_h*f_b) / ((rho_o/rho) * rel_h - rel_h + 1)
        water_depth = utils.clip_min(1e-3, thick - f_b)
        fl = calving_flux_from_depth(gdir, mb_model=mb_model,mb_years=mb_years,k= calving_k,
                                     water_level=water_level, water_depth=water_depth,
                                     calving_law_inv =  calving_law_inv, modelprms = modelprms,
                                     glacier_rgi_table = glacier_rgi_table,hindcast = hindcast,
                                     debug = debug, debug_refreeze = debug_refreeze,
                                     option_areaconstant = option_areaconstant,
                                     inversion_filter = inversion_filter,filesuffix=filesuffix)

        flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
        thick = fl['thick']
        length = len(cls['width']) * cls['dx']
        stretch_dist_p = cfg.PARAMS.get('stretch_dist', 8e3)
        stretch_dist = utils.clip_max(stretch_dist_p,length)
        n_stretch = np.rint(stretch_dist/cls['dx']).astype(int)

        pull_stress = utils.clip_min(0,0.5 * G * (rho * thick**2 -
                                     rho_o * water_depth**2))
        # Define stretch factor and add to driving stress
        stretch_factor = np.zeros(n_stretch)
        for j in range(n_stretch):
            stretch_factor[j] = 2*(j+1)/(n_stretch+1)
        if cls['dx'] > stretch_dist:
            stretch_factor = stretch_dist / cls['dx']
            n_stretch = 1
        stretch_factor = stretch_factor[-1]
        a_pull = (stretch_factor * (pull_stress / stretch_dist))
        a_factor = ((a_pull / (rho*cfg.G*slope*thick)) + 1)
        a_factor = np.nan_to_num(a_factor, nan=1, posinf=1, neginf=1)
        sia_thick = sia_thickness(slope, width, flux, rel_h, a_factor,
                                  glen_a=glen_a, fs=fs)
        is_trap = cl['is_trapezoid'][-1]
        if cl['invert_with_trapezoid']:
            t_lambda = cfg.PARAMS['trapezoid_lambdas']
            min_shape = cfg.PARAMS['mixed_min_shape']
            bed_shape = 4 * sia_thick / cl['width'][-1] ** 2
            is_trap = ((bed_shape < min_shape) & ~ cl['is_rectangular'][-1] &
                       (cl['flux'][-1] > 0)) | is_trap
            if is_trap:
                try:
                    sia_thick = sia_thickness_via_optim(slope, width, flux,
                                                        rel_h, a_factor,
                                                        shape='trapezoid',
                                                        t_lambda=t_lambda,
                                                        glen_a=glen_a, fs=fs)
                except ValueError:
                    # no solution error - we do with rect
                    sia_thick = sia_thickness_via_optim(slope, width, flux,
                                                        rel_h, a_factor,
                                                        shape='rectangular',
                                                        glen_a=glen_a, fs=fs)
#        sia_thick = sia_thickness(slope, width, flux, glen_a=glen_a, fs=fs)
        return fl['thick'] - sia_thick

    # abs_min = optimize.minimize(to_minimize, [1], bounds=((1e-4, 1e4), ),
    #                             tol=1e-1)
    abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e4), ),
                                tol=1e-1)
    print("abs_min is :",abs_min)


    # if not abs_min['success']:
    #     raise RuntimeError('Could not find the absolute minimum in calving '
    #                        'flux optimization: {}'.format(abs_min))
    # if abs_min['fun'] > 0:
    #     # This happens, and means that this glacier simply can't calve
    #     # This is an indicator for physics not matching, often an unrealistic
    #     # slope of free-board
    #     print("############# this glacier simply can't calve #############")
    #     out = calving_flux_from_depth(gdir,mb_model=mb_model,mb_years=mb_years,water_level=water_level)

    #     log.warning('({}) find_inversion_calving_from_any_mb: could not find '
    #                 'calving flux.'.format(gdir.rgi_id))

    #     odf = dict()
    #     odf['calving_flux'] = 0
    #     odf['calving_rate_myr'] = 0
    #     odf['calving_law_flux'] = out['flux']
    #     odf['calving_water_level'] = out['water_level']
    #     odf['calving_inversion_k'] = out['inversion_calving_k']
    #     odf['calving_front_slope'] = slope
    #     odf['calving_front_water_depth'] = out['water_depth']
    #     odf['calving_front_free_board'] = out['free_board']
    #     odf['calving_front_thick'] = out['thick']
    #     odf['calving_front_width'] = out['width']
    #     for k, v in odf.items():
    #         if isinstance(v,np.ndarray):
    #             gdir.add_to_diagnostics(k, v.tolist())   
    #         else:
    #             gdir.add_to_diagnostics(k, v)
    #     return odf

    # # OK, we now find the zero between abs min and an arbitrary high front
    # abs_min = abs_min['x'][0]
    # print('abs_min is',abs_min)
    # try:
    #     opt = optimize.brentq(to_minimize, abs_min, 1e4)
    # except:
    #     print("something is wrong when optimize opt")
    # print("water depth now is :",opt)

    # Shift the water level, if numerical solver can't find a solution with the 
    # initial one. Looking for a solution each shifting it up and down.
    min_wl = -th if th > thick0 else -thick0
    max_wl = th - 1e-3
    success = 0
    water_level_original = water_level
    step = cfg.PARAMS['water_level_step']
    opt_p = None
    opt_m = None
    while abs_min['fun'] > 0 or success == 0:
        abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                    tol=1e-1)
        abs_min0 = abs_min['x'][0]
        try:
            opt_m = optimize.brentq(to_minimize, abs_min0, 1e7)
            success = 1
        except ValueError:
            water_level -= step
            pass
        if water_level <= min_wl:
            water_level = min_wl
            break
    water_level_m = water_level
    success_m = success
 
    water_level = water_level_original
    success = 0
    while abs_min['fun'] > 0 or success == 0:
        abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                    tol=1e-1)
        abs_min0 = abs_min['x'][0]
        try:
            opt_p = optimize.brentq(to_minimize, abs_min0, 1e7)
            success = 1
        except ValueError:
            water_level += step
            pass
        if water_level >= max_wl:
            water_level = max_wl
            break
    water_level_p = water_level
    success_p = success

    # We take the smallest absolute water level. (Except for cases where we 
    # start with a negative water level; see above.)
    if success_p == 1 and np.abs(water_level_p) < np.abs(water_level_m) and not \
       thick0 > (rho_o/(rho_o-rho))*th:
        water_level = water_level_p
        opt = opt_p
    elif success_m == 1:
        water_level = water_level_m
        opt = opt_m
    else:
        water_level = water_level_p
        opt = opt_p

    # Give the flux to the inversion and recompute
    # This is the thick guaranteeing OGGM Flux = Calving Law Flux
    print("opt is:",opt)
    if opt is not None:
        rel_h = opt
        f_b = th - water_level
        print("f_b is:",f_b)
        thick = ((rho_o/rho)*rel_h*f_b) / ((rho_o/rho) * rel_h - rel_h + 1)
        print("thick is:",thick)
        water_depth = utils.clip_min(1e-3, thick - f_b)
        print("water_depth is:",water_depth)
    else:
        # Mostly happening when front becomes very thin...
        log.workflow('({}) inversion routine not working as expected. We just '
                     'take random values and proceed...'.format(gdir.rgi_id))
        f_b = 1e-3
        water_level = th - f_b
        opt = 100
        rel_h = opt
        thick = ((rho_o/rho)*rel_h*f_b) / ((rho_o/rho) * rel_h - rel_h + 1)    
        water_depth = utils.clip_min(1e-3, thick - f_b)
        water_level= th - (thick0-water_depth)

    out = calving_flux_from_depth(gdir, water_level=water_level,water_depth= water_depth,mb_model=mb_model,mb_years=mb_years,  k=calving_k,
                                  calving_law_inv=calving_law_inv,filesuffix=filesuffix)
    # Find volume with water and frontal ablation
    f_calving = out['flux']

    log.info('({}) find_inversion_calving_from_any_mb: found calving flux of '
             '{:.03f} km3 yr-1'.format(gdir.rgi_id, f_calving))
    gdir.inversion_calving_rate = f_calving
    print("The inversion calving rate now is (km3 yr-1):",gdir.inversion_calving_rate)
    with utils.DisableLogger():
        massbalance.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                            mb_years=mb_years,filesuffix=filesuffix)
        prepare_for_inversion(gdir,filesuffix=filesuffix)
        mass_conservation_inversion(gdir, water_level=water_level,
                                    glen_a=glen_a, fs=fs, min_rel_h=opt,filesuffix=filesuffix)
        # try:
        #     workflow.calibrate_inversion_from_consensus(gdir, apply_fs_on_mismatch=True,filter_inversion_output =True,volume_m3_reference=None)
        # except:
        #     print("Something is wrong with the function calibrate_inversion_from_consensus")
        #     print(traceback.format_exc())

    out = calving_flux_from_depth(gdir, water_level=water_level,water_depth=water_depth, k=calving_k,
                                  mb_model=mb_model,mb_years=mb_years,
                                  calving_law_inv=calving_law_inv,filesuffix=filesuffix)

    fl = gdir.read_pickle('inversion_flowlines',filesuffix=filesuffix)[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])
    
    log.workflow('({}) found frontal thickness, water depth, freeboard, water '
                 'level of {}, {}, {}, {}'.format(gdir.rgi_id, out['thick'],
                 out['water_depth'],out['free_board'], out['water_level']))
    log.workflow('({}) calving (law) flux of {} ({})'.format(gdir.rgi_id,
                 f_calving, out['flux']))
    # Store results
    # Check the  type of the variables, should be list, not array
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_rate_myr'] = f_calving * 1e9 / (out['thick'] * out['width'])
    odf['calving_law_flux'] = out['flux']
    odf['calving_water_level'] = out['water_level']
    odf['calving_inversion_k'] = out['inversion_calving_k']
    odf['calving_front_slope'] = slope
    odf['calving_front_water_depth'] = out['water_depth']
    odf['calving_front_free_board'] = out['free_board']
    odf['calving_front_thick'] = out['thick']
    odf['calving_front_width'] = out['width']
    try:
        for k, v in odf.items():
            # convert Numpy array to Python list
            if isinstance(v,np.ndarray):
                print("Key :",k,"Value :",v)
                v = v.tolist()
            gdir.add_to_diagnostics(k, v,filesuffix=filesuffix)
    except:
        print(traceback.format_exc())

    return odf

