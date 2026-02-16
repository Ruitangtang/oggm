"""
Idealized bed topography functions

Author: Lizz Ultee, Fabien Maussion & the OGGM developers

"""

import numpy as np
from oggm.core.flowline import RectangularBedFlowline


def bu_tidewater_bed(gridsize=200, gridlength=6e4, widths_m=600,
                     b_0=260, alpha=0.017, b_1=350, x_0=4e4, sigma=1e4,
                     water_level=0, split_flowline_before_water=None,
                     with_spin_up=None):

    # Bassis & Ultee bed profile
    dx_meter = gridlength / gridsize
    x = np.arange(gridsize+1) * dx_meter
    bed_h = b_0 - alpha * x + b_1 * np.exp(-((x - x_0) / sigma)**2)
    bed_h += water_level

    if with_spin_up is not None:
        db = bed_h[0] - bed_h[1]
        bed_s = np.arange(with_spin_up)[::-1] * db + bed_h[0] + db
        bed_h = np.append(bed_s, bed_h)

    surface_h = bed_h
    widths = surface_h * 0. + widths_m / dx_meter

    if with_spin_up is not None:
        fls = [RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[:with_spin_up],
                                      bed_h=bed_h[:with_spin_up],
                                      widths=widths[:with_spin_up]),
               RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[with_spin_up:],
                                      bed_h=bed_h[with_spin_up:],
                                      widths=widths[with_spin_up:]),
               ]
        fls[0].set_flows_to(fls[1], check_tail=False, to_head=True)
        return fls
    elif split_flowline_before_water is not None:
        bs = np.min(np.nonzero(bed_h < 0)[0]) - split_flowline_before_water
        fls = [RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[:bs],
                                      bed_h=bed_h[:bs],
                                      widths=widths[:bs]),
               RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[bs:],
                                      bed_h=bed_h[bs:],
                                      widths=widths[bs:]),
               ]
        fls[0].set_flows_to(fls[1], check_tail=False, to_head=True)
        return fls
    else:
        return [
            RectangularBedFlowline(dx=1, map_dx=dx_meter, surface_h=surface_h,
                                   bed_h=bed_h, widths=widths)]