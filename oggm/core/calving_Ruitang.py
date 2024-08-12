# Builtins
import logging
import copy
from functools import partial


# External libs
import numpy as np
import pandas as pd
import traceback
import sys

# Optional libs
try:
    from skimage import measure
except ImportError:
    pass

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError
from oggm.core.inversion import find_sia_flux_from_thickness
from oggm.core.flowline import FlowlineModel, flux_gate_with_build_up

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
        cfg.PARAMS['ocean_density'] = 1028

    # Stretch distance in hydrostatic pressure balance
    # calculations for terminal water-terminating cliffs
    # (in meters) - See Malles et al.
    # (the current value of 8000m might be too high)
    if 'max_calving_stretch_distance' not in cfg.PARAMS:
        cfg.PARAMS['max_calving_stretch_distance'] = 8000

def k_calving_law(model, flowline, last_above_wl):
    """Compute calving from the model state using the k-calving law.

    Currently this still assumes that the model has an attribute
    called "calving_k", which might be changed in the future.

    Parameters
    ----------
    model : oggm.core.flowline.FlowlineModel
        the model instance calling the function
    flowline : oggm.core.flowline.Flowline
        the instance of the flowline object on which the calving law is called
    last_above_wl : int
        the index of the last pixel above water (in case you need to know
        where it is).
    """
    h = flowline.thick[last_above_wl]
    d = h - (flowline.surface_h[last_above_wl] - model.water_level)
    k = model.calving_k
    q_calving = k * d * h * flowline.widths_m[last_above_wl]
    return q_calving


def fa_sermeq_speed_law(model,last_above_wl, v_scaling=1, verbose=False,
                     tau0=1.5, variable_yield=None, mu=0.01,
                     trim_profile=1):
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
    def tau_y(tau0=1.3, variable_yield=None, bed_elev=None, thick=None, mu=0.01):
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
    
    # u_stag[-1] is the main flowline
    velocity_m = model.u_stag[-1]*cfg.SEC_IN_YEAR
    print('velocity_m is', velocity_m)
    x_m = flowline.dis_on_line*flowline.map_dx
    print('x_m in fa_sermq_law is :',x_m)

    # gdir : py:class:`oggm.GlacierDirectory`
    #     the glacier directory to process
    # fls = model.gdir.read_pickle('model_flowlines')
    # mbmod_fl = massbalance.MultipleFlowlineMassBalance(model.gdir, fls=fls, use_inversion_flowlines=True,
    #                                                    mb_model_class=MonthlyTIModel)
    #mb_annual=model.mb_model.get_annual_mb(heights=surface_m, fl_id=-1, year=model.yr, fls=model.fls)
    #  should call the monthly function
    mb_annual=model.mb_model.get_monthly_mb(heights=surface_m, fl_id=-1, year=model.yr, fls=model.fls)

    print("mb_annual is (m ice per second):",mb_annual,"in year",model.yr,"Actually is monthly output")
    Terminus_mb = mb_annual*cfg.SEC_IN_YEAR
    print("Terminus mass balance is (m per year):",Terminus_mb)
    # slice up to index+1 to include the last nonzero value
    # profile: NDarray
    #     The current profile (x, surface, bed,width) as calculated by the base model
    #     Unlike core SERMeQ, these should be DIMENSIONAL [m].
    profile=(x_m[:last_above_wl+1],
                 surface_m[:last_above_wl+1],
                 bed_m[:last_above_wl+1],width_m[:last_above_wl+1])
    print("bed_h of the flowline using to do calving is :", bed_m[:last_above_wl+1])
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
    print("the surface at the terminus is (m a.s.l.) :",se_terminus)
    print("the bed at the terminus is (m a.s.l.) :",bed_terminus)
    h_terminus = se_terminus - bed_terminus
    width_terminus = profile[3][last_index]
    tau_y_terminus = tau_y(tau0=tau0, bed_elev=bed_terminus, thick=h_terminus, variable_yield=variable_yield)
    print('tau_y_terminus in fa_sermeq_speed_law is:',tau_y_terminus)  
    Hy_terminus = balance_thickness(yield_strength=tau_y_terminus, bed_elev=bed_terminus)
    print('Hy_terminus in fa_sermeq_speed_law is:',Hy_terminus)  
    if isinstance(model_velocity, (int, float)):
        U_terminus = model_velocity
        U_adj = model_velocity
    else:
        U_terminus = model_velocity[last_index]  ## velocity, assuming last point is terminus
        U_adj = model_velocity[last_index - 1]
    print(f"the U terminus and adj are (m a-1): {U_terminus} and {U_adj}")

    ## Ice thickness and yield thickness at adjacent point
    se_adj = profile[1][last_index - 1]
    bed_adj = profile[2][last_index - 1]
    print("the surface at the grid backbefore the terminus (adj) (m a.s.l.) is :" ,se_adj)
    print("the bed at the grid backbefore the terminus (adj) (m a.s.l.) is :" ,bed_adj)
    H_adj = se_adj - bed_adj
    tau_y_adj = tau_y(tau0=tau0, bed_elev=bed_adj, thick=H_adj, variable_yield=variable_yield)
    print('tau_y_adj in fa_sermeq_speed_law is:',tau_y_adj)
    Hy_adj = balance_thickness(yield_strength=tau_y_adj, bed_elev=bed_adj)
    print('Hy_adj in fa_sermeq_speed_law is:',Hy_adj)
    # Gradients
    dx_term = profile[0][last_index] - profile[0][last_index - 1]  ## check grid spacing close to terminus
    if dx_term <= 0.0 :
        raise RuntimeError('DX_TERM IS LESS THEN ZERO')
    dHdx = (h_terminus - H_adj) / dx_term
    # if dHdx >= 0.0 :
    #     raise RuntimeError('DHDX_TERM IS LESS THEN ZERO')
    dHydx = (Hy_terminus - Hy_adj) / dx_term
    # if dHydx <= 0.0 :
    #     raise RuntimeError('DHYDX_TERM IS LESS THEN ZERO')
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
        dUdx = (U_terminus - U_adj) / dx_term  ## velocity gradient
        ## Group the terms
        dLdt_numerator = terminus_mb - (h_terminus * dUdx) - (U_terminus * dHdx)
        dLdt_denominator = dHydx - dHdx  ## TODO: compute dHydx
        dLdt_viscoplastic = dLdt_numerator / dLdt_denominator
        # if dLdt_denominator < 0:
        #     raise RuntimeError('DHYDX-DHDX IS LESS THEN ZERO')
        # elif dLdt_denominator == 0:
        #     raise RuntimeError('DHYDX-DHDX IS ZERO')
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
        print('Checking dLdt: terminus_mb = {}. \n H dUdx = {}. \n U dHdx = {}.'.format(terminus_mb, dUdx * h_terminus,
                                                                                        U_terminus * dHdx))
        print('Denom: dHydx = {} \n dHdx = {}'.format(dHydx, dHdx))
        print('Viscoplastic dLdt={}'.format(dLdt_viscoplastic))
        print('Terminus surface mass balance ma= {}'.format(terminus_mb))
        print('Sermeq frontal ablation ma={}'.format(fa_viscoplastic))
    else:
        pass
    return SQFA




class CalvingFluxBasedModelRt(FlowlineModel):
    """This is the version currently in progress.

    Written by Fabien based on Jan's code. Code is more efficient
    but does not exactly do the same and does not work at the moment.
    """
    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=100,
                 do_kcalving=None, calving_k=None, calving_law=fa_sermeq_speed_law,
                 variable_yield=None, calving_use_limiter=None, calving_limiter_frac=None,
                 water_level=None,mb_elev_feedback='monthly',
                 **kwargs):
        """Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overridden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        calving_law : func
             option to use another calving law. This is a temporary workaround
             to test other calving laws, and the system might be improved in
             future OGGM versions.
             1. k-calving law
             2. fa_sermq_speed_law
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `flowline_model_run` does for you)
        """
        super(CalvingFluxBasedModelRt, self).__init__(flowlines, mb_model=mb_model,
                                                        y0=y0, glen_a=glen_a, fs=fs,
                                                        inplace=inplace,
                                                        water_level=water_level,mb_elev_feedback=mb_elev_feedback,
                                                        **kwargs)

        # Initialize the parameters
        initialize_calving_params()

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.min_dt = min_dt
        self.cfl_number = cfl_number

        # Calving params
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.do_calving = do_kcalving and self.is_tidewater
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']
        self.calving_k = calving_k / cfg.SEC_IN_YEAR
        self.calving_law = calving_law
        if calving_limiter_frac is None:
            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
        if variable_yield is None:
            variable_yield = cfg.PARAMS['variable_yield']
        self.variable_yield = variable_yield
        if calving_limiter_frac > 0:
            raise NotImplementedError('calving limiter other than 0 not '
                                      'implemented yet')
        self.calving_limiter_frac = calving_limiter_frac
        # Flux gate
        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
        self.flux_gate_m3_since_y0 = 0.
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            flux_gate_thickness = utils.tolist(flux_gate_thickness,
                                               length=len(self.fls))
            self.flux_gate = []
            for fl, fgt in zip(self.fls, flux_gate_thickness):
                # We set the thickness to the desired value so that
                # the widths work ok
                fl = copy.deepcopy(fl)
                fl.thick = fl.thick * 0 + fgt
                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
                if slope == 0:
                    raise ValueError('I need a slope to compute the flux')
                flux = find_sia_flux_from_thickness(slope,
                                                    fl.widths_m[0],
                                                    fgt,
                                                    shape=fl.shape_str[0],
                                                    glen_a=self.glen_a,
                                                    fs=self.fs)
                self.flux_gate.append(flux)

        # convert the floats to function calls
        for i, fg in enumerate(self.flux_gate):
            if fg is None:
                continue
            try:
                # Do we have a function? If yes all good
                fg(self.yr)
            except TypeError:
                # If not, make one
                self.flux_gate[i] = partial(flux_gate_with_build_up,
                                            flux_value=fg,
                                            flux_gate_yr=(flux_gate_build_up +
                                                          self.y0))
        # Special output
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.u_stag = []
        self.ud_stag = []  # deformation velocity (for diagnostics)
        self.us_stag = []  # sliding velocity (for diagnostics)
        self.flux_stag = []
        self.trib_flux = []
        self.water_depth_stag = []  # this is a constant, we compute it below
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add a fake grid point at the end of tributaries
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.ud_stag.append(np.zeros(nx+1))
            self.us_stag.append(np.zeros(nx+1))
            self.flux_stag.append(np.zeros(nx+1))
            # Staggered water depth (constant)
            water_depth_stag = np.zeros(nx + 1)
            depth = utils.clip_min(0,self.water_level - fl.bed_h)
            water_depth_stag[1:-1] = (depth[0:-1] + depth[1:]) / 2.
            water_depth_stag[[0, -1]] = depth[[0, -1]]
            self.water_depth_stag.append(water_depth_stag)

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # Pick the containers
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            ud_stag = self.ud_stag[fl_id]
            us_stag = self.us_stag[fl_id]
            flux_gate = self.flux_gate[fl_id]
            water_depth_stag = self.water_depth_stag[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter
            #water_depth = fl.water_depth
            water_depth = utils.clip_min(0,self.water_level - fl.bed_h)
            calving_flux = 0.

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered velocity
            # -> depends on calving
            if self.do_calving:
                ice_in_water = (fl.bed_h < self.water_level) & (fl.thick > 0)
            else:
                ice_in_water = False

            N = self.glen_n
            calving_is_happening = False

            # First determine if calving (or sliding) might be happening
            if self.do_calving and np.any(ice_in_water):

                # Here we have to do two things:
                # - change the sliding where the bed is below water
                # - decide on whether calving is happening or not

                rho_ocean = cfg.PARAMS['ocean_density']
                eff_water_depth = (rho_ocean / self.rho) * water_depth

                # above_wl -> above floatation
                ice_above_wl = ((fl.bed_h < self.water_level) & (fl.thick >= eff_water_depth))

                if np.any(ice_above_wl):
                    last_above_wl = np.where(ice_above_wl)[0][-1]
                else:
                    # Sometimes when glaciers are advancing there is a
                    # bit of ice into the water, but it's not above water.
                    # Let's use that instead
                    last_above_wl = np.where(ice_in_water)[0][-1]

                # Don't go further than the end of the domain
                if last_above_wl >= len(fl.bed_h) - 2:
                    last_above_wl = len(fl.bed_h) - 2

                # Check that the "last_above_wl" is not just the last in a
                # "lake" (over-deepening) which is followed by land again.
                # If after last_above_wl we still have ice, we don't calve.
                calving_is_happening = not fl.thick[last_above_wl + 1] > 0

                # Determine water depth at the front
                h = fl.thick[last_above_wl]
                d = h - (fl.surface_h[last_above_wl] - self.water_level)

                # Force the staggered grid to these values (Fabi asks: why?)
                # thick_stag[last_above_wl + 1] = h
                # water_depth_stag[last_above_wl + 1] = d

                # Compute net hydrostatic force at the front. One could think
                # about incorporating ice mélange / sea ice here as an
                # additional backstress term.
                # (And also in the frontal ablation formulation below.)
                stretch_dist = 1
                if calving_is_happening:

                    # Calculate the additional (pull) force
                    pull_last = 0.5 * G * (self.rho * h ** 2 - rho_ocean * d ** 2)
                    if pull_last < 0:
                        pull_last = 0

                    # Determine distance over which above force is distributed
                    max_dist = cfg.PARAMS['max_calving_stretch_distance']
                    first_ice = np.where(fl.thick > 0)[0][0]
                    glacier_len = (last_above_wl - first_ice) * dx
                    if glacier_len < dx:
                        glacier_len = dx

                    stretch_dist = glacier_len if glacier_len < max_dist else max_dist
                    n_stretch = np.rint(stretch_dist / dx).astype(int)

                    # Define stretch factor and add to driving stress
                    stretch_factor = np.arange(1, n_stretch + 2) * 2 / (n_stretch + 1)
                    stretch_last = last_above_wl + 2  # because last is excluded
                    stretch_first = (last_above_wl + 2) - n_stretch

                    # Take slope for stress calculation at boundary grid cell
                    # as the mean over the stretched distance (see above)
                    if stretch_first != stretch_last - 1:
                        avg_sl = np.mean(slope_stag[stretch_first - 1: stretch_last - 1])
                        slope_stag[last_above_wl + 1] = avg_sl

                # OK now we compute the deformation stress, which might
                # be slightly different at the front if calving is happening
                stress = self.rho * G * slope_stag * thick_stag

                # Add "stretching stress" to basal shear/driving stress
                if calving_is_happening:
                    stress[stretch_first:stretch_last+1] += stretch_factor * (pull_last / stretch_dist)

                # Compute velocities
                # Deformation is the usual formulation with changed stress
                ud_stag[:] = thick_stag * stress ** N * self._fd

                # Sliding is increased where there is water
                # Determine height above buoyancy
                eff_water_depth_stag = (rho_ocean / self.rho) * water_depth_stag
                # Avoid dividing by zero where thick equals water depth
                z_a_b = utils.clip_min(thick_stag - eff_water_depth_stag, 0.01)
                z_a_b[thick_stag == 0] = 1  # Stress is zero there
                us_stag[:] = (stress ** N / z_a_b) * self.fs

                # Force velocity beyond grounding line to be zero in order to
                # prevent shelf dynamics. This might accumulate too much volume
                # in the last_above_wl+1 grid cell, which we deal with in the
                # calving scheme below
                if calving_is_happening:
                    us_stag[last_above_wl + 2:] = 0
                    ud_stag[last_above_wl + 2:] = 0

                u_stag[:] = ud_stag + us_stag

                if calving_is_happening:
                    # For the flux out of the last grid cell, the staggered section
                    # is set to the cross-section of the calving front, as we are
                    # dealing with a terminal cliff.
                    section_stag[last_above_wl + 1] = section[last_above_wl]

                    # We calculate the calving flux here to be consistent with
                    # mass balance which is also computed before mass
                    # redistribution
                    k = self.calving_k
                    w = fl.widths_m[last_above_wl]
                    print("calving_k before the fa_sermeq_speed_law is :",k)
                    if self.calving_law == fa_sermeq_speed_law:
                        print("before calving")
                        print("model.yr is :",self.yr)
                        try:
                            # Transit the unit of tau0 to Pa, based on the equation self.calving_k= calving_k/cfg.SEC_IN_YEAR
                            # tau0 = self.calving_k * cfg.SEC_IN_YEAR
                            s_fa = self.calving_law(self, last_above_wl,v_scaling = 1, verbose = True,tau0 = k*cfg.SEC_IN_YEAR,
                                                variable_yield=self.variable_yield, mu = 0.01,trim_profile = 0)
                            calving_flux = s_fa ['Sermeq_fa']*s_fa['Thickness_termi']*s_fa['Width_termi']/cfg.SEC_IN_YEAR


                        except RuntimeError:
                            traceback.print_exception(*sys.exc_info())
                    else:
                        calving_flux = self.calving_law(self, fl, last_above_wl)

                    if calving_flux < 0:
                        calving_flux = 0
                    print("calving_flux is (m3 s-1):",calving_flux)
            else:

                # Simplest case (no water): velocity is deformation + sliding
                rhogh = (self.rho * G * slope_stag)**N
                ud_stag[:] = (thick_stag**(N + 1)) * self._fd * rhogh

                # Temporary check for above
                stress = self.rho * G * slope_stag * thick_stag
                vel = thick_stag * stress ** N * self._fd
                assert np.allclose(ud_stag, vel)

                us_stag[:] = (thick_stag**(N - 1)) * self.fs * rhogh

                # Temporary check for above
                z_a_b = thick_stag
                z_a_b[thick_stag == 0] = 1
                vel = (stress ** N / z_a_b) * self.fs
                assert np.allclose(us_stag, vel)
                u_stag[:] = ud_stag + us_stag

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                print("maxu is (m s-1) :",maxu)
                if maxu > cfg.FLOAT_EPS:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s. Happening at '
                            'simulation year {:.1f}, fl_id {}, '
                            'bin_id {} and max_u {:.3f} m yr-1.'
                            ''.format(cfl_dt, self.min_dt, self.yr, fl_id,
                                      np.argmax(np.abs(u_stag)),
                                      maxu * cfg.SEC_IN_YEAR))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None

            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # We prevent MB processes below water, since this should be
            # part of calving (in theory)
            if calving_is_happening:
                mb[fl.surface_h < 0] = 0

            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # --- The rest is for calving only ---
            self.calving_rate_myr = 0.

            # If tributary, do the things below only if we are not transferring mass
            if is_trib and flx_stag[-1] > 0:
                continue

            # No need to do calving in these cases either
            if not self.do_calving or not fl.has_ice():
                continue

            # We do calving only if the last glacier bed pixel is below water
            # (this is to avoid calving elsewhere than at the front)
            # TODO - do we really want this?
            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
                continue

            section = fl.section

            # If there is only ice below water, we just remove it
            # (should be super rare?)
            if np.all(fl.surface_h[fl.thick > 0] < self.water_level):
                iceberg_calving_m3 = np.sum(section) * dx
                self.calving_m3_since_y0 += iceberg_calving_m3
                section[:] = 0
                fl.section = section
                fl.calving_bucket_m3 = 0
                continue

            # Remove detached bodies of ice in the water, which can happen as
            # a result of dynamics + surface melt
            # Detached means bed below water and some ice thickness,
            # but not touching any other areas. That's perfect for labelling.
            just_ice = fl.thick > 0
            ice_in_water = just_ice & (fl.bed_h < self.water_level)
            just_ice_labels = measure.label(just_ice)
            if just_ice_labels.max() > 1:
                # OK we have disconnections. Is one of them fully in water?
                for label in range(2, just_ice_labels.max() + 1):
                    is_detached = just_ice_labels == label
                    if not np.all(ice_in_water[is_detached]):
                        # That's still connected to land, ignore
                        continue
                    if is_detached.sum() > 5:
                        # Arbitrary threshold. Do we really want to un-ground
                        # that much ice in one step?
                        continue

                    iceberg_calving_m3 = np.sum(section[is_detached]) * dx
                    self.calving_m3_since_y0 += iceberg_calving_m3
                    self.calving_rate_myr += (iceberg_calving_m3 / dt /
                                              section[last_above_wl] *
                                              cfg.SEC_IN_YEAR)

                    section[is_detached] = 0
                    fl.section = section  # recompute the other vars

            if not calving_is_happening:
                continue

            # # Make sure that we have a smooth front. Necessary due to inhibiting
            # # ice flux beyond the grounding line in the dynamics above
            # section = fl.section
            # while ((fl.surface_h[last_above_wl] > fl.surface_h[last_above_wl-1])
            #        and fl.thick[last_above_wl] > 0) and last_above_wl > 0:
            #     old_thick = fl.thick[last_above_wl]
            #     old_sec = fl.section[last_above_wl]
            #     new_thick = (old_thick - (fl.surface_h[last_above_wl] -
            #                               fl.surface_h[last_above_wl-1]))
            #     fl.thick[last_above_wl] = new_thick
            #     diff_sec = old_sec - fl.section[last_above_wl]
            #     section[last_above_wl+1] += diff_sec
            #     section[last_above_wl] -= diff_sec
            #     fl.section = section
            #     section = fl.section
            #     if ((fl.bed_h[last_above_wl+1] < self.water_level) &
            #         (fl.thick[last_above_wl+1] >= (rho_ocean / self.rho) *
            #          water_depth[last_above_wl+1])):
            #         last_above_wl += 1
            #     else:
            #         break

            # OK, we're really calving
            q_calving = calving_flux * dt

            # Remove ice below flotation beyond the first grid cell after the
            # grounding line. That cell is the "advance" bucket, meaning it can
            # contain ice below flotation.
            add_calving = np.sum(section[last_above_wl+2:]) * dx

            # Add to the bucket and the diagnostics. As we calculated the
            # calving flux from the glacier state at the start of the time step,
            # we do not add to the calving bucket what is already removed by the
            # flotation criterion to avoid double counting.
            self.calving_m3_since_y0 += utils.clip_min(q_calving, add_calving)
            fl.calving_bucket_m3 += utils.clip_min(0, q_calving - add_calving)
            self.calving_rate_myr += (utils.clip_min(q_calving, add_calving) /
                                      fl.section[last_above_wl] / dt *
                                      cfg.SEC_IN_YEAR)
            section[last_above_wl+2:] = 0

            # This is what remains to be removed by the calving bucket.
            to_remove = section[last_above_wl+1] * dx
            if 0 < to_remove < fl.calving_bucket_m3:
                # This is easy, we remove everything
                section[last_above_wl+1] = 0
                fl.calving_bucket_m3 -= to_remove
            elif to_remove > 0:
                # We can only remove part of it
                section[last_above_wl+1] = ((to_remove - fl.calving_bucket_m3) / dx)
                fl.calving_bucket_m3 = 0

            vol_last = section[last_above_wl] * dx
            while fl.calving_bucket_m3 >= vol_last and \
                    fl.bed_h[last_above_wl] < self.water_level:
                fl.calving_bucket_m3 -= vol_last
                section[last_above_wl] = 0

                # OK check if we need to continue (unlikely)
                last_above_wl -= 1
                if np.abs(last_above_wl) <= len(fl.bed_h):
                    vol_last = section[last_above_wl] * dx
                else:
                    fl.calving_bucket_m3 = 0
                    break

            # We update the glacier with our changes
            fl.section = section

        # Next step
        self.t += dt
        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Velocities in OGGM's FluxBasedModel are sometimes subject to
        numerical instabilities. To deal with the issue, you can either
        set a smaller ``PARAMS['cfl_number']`` (e.g. 0.01) or smooth the
        output a bit, e.g. with ``df.rolling(5, center=True, min_periods=1).mean()``

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
            - surface_ice_velocity: m per second (corrected for surface - simplified)
        """
        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
        var = self.ud_stag[fl_id]
        df['deformation_velocity'] = (var[1:nx+1] + var[:nx])/2
        var = self.us_stag[fl_id]
        df['sliding_velocity'] = (var[1:nx+1] + var[:nx])/2

        # Surface vel is deformation corrected by factor + sliding
        var = self.ud_stag[fl_id]
        ud = (var[1:nx+1] + var[:nx])/2
        var = self.us_stag[fl_id]
        us = (var[1:nx+1] + var[:nx])/2
        df['surface_ice_velocity'] = ud * self._surf_vel_fac + us

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id]

        return df