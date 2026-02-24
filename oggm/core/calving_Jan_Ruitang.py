# Builtins
import logging
import copy
from functools import partial
import warnings

# External libs
import numpy as np
import pandas as pd
import traceback
import sys

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError
from oggm.core.inversion import find_sia_flux_from_thickness
#from oggm.core.flowline import FlowlineModel, flux_gate_with_build_up

# Constants
from oggm.cfg import G

# Module logger
log = logging.getLogger(__name__)


def initialize_calving_params():
    """Initialize the parameters for the calving model.

    This should be part of params.cfg but this is still
    sandboxed...
    """

    # ocean water density in kg m-3; should be >= ice density
    # for lake-terminating glaciers this could be changed to
    # 1000 kg m-3
    if 'ocean_density' not in cfg.PARAMS:
        cfg.PARAMS['ocean_density'] = 1020

    # Stretch distance in hydrostatic pressure balance
    # calculations for terminal water-terminating cliffs
    # (in meters) - See Malles et al.
    # (the current value of 8000m might be too high)
    if 'max_calving_stretch_distance' not in cfg.PARAMS:
        cfg.PARAMS['max_calving_stretch_distance'] = 8000





def fa_sermeq_speed_law(model,last_above_wl, mb_current = None,v_scaling=1, verbose=False,
                     tau0=1.5, variable_yield=None, mu=0.01,
                     trim_profile=1,mb_elev_feedback='monthly'):
    """
    This function is used to calculate frontal ablation given ice speed forcing,
    for lake-terminating and tidewater glaciers

    @author: Ruitang Yang & Lizz Ultee

    Authors: Ruitang Yang & Lizz Ultee
    Parameters
    ----------

    model : oggm.core.flowline.FlowlineModel
        the model instance calling the function
    flowline : oggm.core.flowline.Flowline
        the instance of the flowline object on which the calving law is called
    fl_id : float, optional
        the index of the flowline in the fls array (might be ignored by some MB models)
    last_above_wl : int
        the index of the last pixel above water (in case you need to know
        where it is).
    mb_current : float, optional, array
        the current mass balance (has been computed by the model), along the flowline
    v_scaling: float
        velocity scaling factor, >0, default is 1
    Terminus_mb : array
        Mass balance along the flowline or nearest the terminus [m/a]. Default None,the unit meter of ice per year 
    verbose: Boolean, optional
        Whether to print component parts for inspection.  Default False.

    tau0: float, optional
        This glacier's yield strength [Pa]. Default is 150 kPa.
    yield_type: str, optional
        'constant' or 'variable' (Mohr-Coulomb) yielding. Default is constant.
    mu: float, optional
        Mohr-Coulomb cohesion, a coefficient between 0 and 1. Default is 0.01.
        Only used if we have variable yield

    trim_profile: int, optional
        How many grid cells at the end of the profile to ignore.  Default is 1.
        If the initial profile is set by k-calving (as in testing) there can be a
        weird cliff shape with very thin final grid point and large velocity gradient

    mb_elev_feedback : str, default: 'monthly' (dynamic step is monthly) ; 'annual' (dynamic step is annual)

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
        Surface mass balance at the terminus [m/a] m ice per year
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
    water_level = model.water_level
    print("water_level in the fa_sermeq_speed_law at the start is (m) :",water_level)
    if variable_yield is not None and not variable_yield:
        variable_yield = None
    print("variable_yield is ", variable_yield)
    

    # ---------------------------------------------------------------------------
    # the yield strength
    def tau_y(tau0=1.5, variable_yield=None, bed_elev=None, thick=None, mu=0.01):
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
            try:
                # if bed_elev < 0:
                #     D = -1 * bed_elev  # Water depth D the nondim bed topography value when Z<0
                # else:
                #     D = 0
                D = utils.clip_min(0,water_level - bed_elev)
            except:
                print('You must set a bed elevation and ice thickness to use variable yield strength. Using constant yeild instead')
                ty = tau1
            N = RHO_ICE * G * thick - RHO_SEA * G * D # Normal stress at bed
            #convert to Pa
            
            ty = tau1 + mu * N
        else:  # assume constant if not set
            ty = tau1
        return ty


    # ---------------------------------------------------------------------------
    # calculate the yield ice thickness

    def balance_thickness(yield_strength, bed_elev):
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
            The ice thickness for stress balance at the terminus. [units]
        """
        # if bed_elev < 0:
        #     D = -1 * bed_elev
        # else:
        #     D = 0
        D = utils.clip_min(0,water_level - bed_elev)
        return (2 * yield_strength / (RHO_ICE * G)) + np.sqrt(
            (RHO_SEA * (D ** 2) / RHO_ICE) + ((2 * yield_strength / (RHO_ICE * G)) ** 2))
        # TODO: Check on exponent on last term.  In Ultee & Bassis 2016, this is squared, but in Ultee & Bassis 2020 supplement, it isn't.

    # ---------------------------------------------------------------------------
    # calculate frontal ablation based on the ice thickness, speed at the terminus
    fls=model.fls
    flowline=fls[-1]
    surface_m = flowline.surface_h
    bed_m = flowline.bed_h
    width_m = flowline.widths_m
    length_m = flowline.length_m
    
    velocity_m = model.u_stag[-1]*cfg.SEC_IN_YEAR
    x_m = flowline.dis_on_line*flowline.map_dx
    #  should call the monthly function
    if mb_current is None:
        if mb_elev_feedback=='monthly':
            mb_annual=model.mb_model.get_monthly_mb(heights=surface_m, fl_id=-1, year=model.yr, fls=model.fls)
        else:
            mb_annual=model.mb_model.get_annual_mb(heights=surface_m, fl_id=-1, year=model.yr, fls=model.fls)

    else:
        mb_annual=mb_current

    Terminus_mb = mb_annual*cfg.SEC_IN_YEAR
    # slice up to index+1 to include the last nonzero value
    # profile: NDarray
    #     The current profile (x, surface, bed,width) as calculated by the base model
    #     Unlike core SERMeQ, these should be DIMENSIONAL [m].
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
    last_index = -1 * (trim_profile + 1)
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
    ## Ice thickness and yield thickness nearest the terminuss
    se_terminus = profile[1][last_index]
    bed_terminus = profile[2][last_index]
    h_terminus = se_terminus - bed_terminus
    width_terminus = profile[3][last_index]
    tau_y_terminus = tau_y(tau0=tau0, bed_elev=bed_terminus, thick=h_terminus, variable_yield=variable_yield) 
    Hy_terminus = balance_thickness(yield_strength=tau_y_terminus, bed_elev=bed_terminus)
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
    tau_y_adj = tau_y(tau0=tau0, bed_elev=bed_adj, thick=H_adj, variable_yield=variable_yield)
    #print('tau_y_adj in fa_sermeq_speed_law is:',tau_y_adj)
    Hy_adj = balance_thickness(yield_strength=tau_y_adj, bed_elev=bed_adj)
    # Gradients
    dx_term = profile[0][last_index] - profile[0][last_index - 1]  ## check grid spacing close to terminus
    if dx_term <= 0.0 :
        raise RuntimeError('DX_TERM IS LESS THEN ZERO')
    dHdx = (h_terminus - H_adj) / dx_term
    dHydx = (Hy_terminus - Hy_adj) / dx_term
    if np.isnan(U_terminus) or np.isnan(U_adj):
        dUdx = np.nan  ## velocity gradient
        ## Group the terms
        dLdt_numerator = np.nan
        dLdt_denominator = np.nan  ## TODO: compute dHydx
        dLdt_viscoplastic = np.nan
        fa_viscoplastic = np.nan  ## frontal ablation rate
    else:
        # Gradients
        # dx_term = profile[0][last_index] - profile[0][last_index - 1]  ## check grid spacing close to terminus 
        # dHdx = (h_terminus - H_adj) / dx_term
        # dHydx = (Hy_terminus - Hy_adj) / dx_term
        dUdx = (U_terminus - U_adj) / dx_term  ## velocity gradient
        ## Group the terms
        dLdt_numerator = terminus_mb - (h_terminus * dUdx) - (U_terminus * dHdx)
        dLdt_denominator = dHydx - dHdx  ## TODO: compute dHydx
        dLdt_viscoplastic = dLdt_numerator / dLdt_denominator
        # check the length change rate should be constrained by smaller than the original length
        if abs(dLdt_viscoplastic) >= length_m:
            print("The absolute length change rate is larger than or equal to the original length, please check the input data and model results")
            # set the length change rate as nan
            dLdt_viscoplastic = np.nan

        U_calving = U_terminus - dLdt_viscoplastic  ## frontal ablation rate
        # add the constraint that the frontal ablation should be non-negative and not infinite by constrain terminus velocity, no more than 5000 m/a
        # we just set the frontal ablation as zero when the terminus velocity is larger than 5000 m/a, which is not realistic for most of the tidewater glaciers,
        # except for some surge type glaciers # TODO: need to be improved in future for surge type glaciers
        # if U_terminus > 5000:
        #     print("The terminus velocity is larger than 5000 m/a, which is not realistic for most of the tidewater glaciers, please check the model results and input data, and consider to set the frontal ablation as zero for this case")
        #     U_calving=0
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
            'Sermeq_fa': fa_viscoplastic,
            'mb_current': mb_annual}
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
        print('Checking dLdt: terminus_mb = {}. \n H dUdx = {}. \n U dHdx = {}.'.format(terminus_mb, dUdx * h_terminus,
                                                                                        U_terminus * dHdx))
        print('Denom: dHydx = {} \n dHdx = {}'.format(dHydx, dHdx))
        print('Viscoplastic dLdt={}'.format(dLdt_viscoplastic))
        print('Terminus surface mass balance ma= {}'.format(terminus_mb))
        print('Sermeq frontal ablation ma={}'.format(fa_viscoplastic))
        print('current_mb={}'.format(mb_annual))
    else:
        pass
    return SQFA



