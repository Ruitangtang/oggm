
"""Useful plotting functions"""
import os
import functools
import logging
from collections import OrderedDict
import itertools
import textwrap
import xarray as xr

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpg

try:
    import salem
except ImportError:
    pass

OGGM_CMAPS = dict()

from oggm.core.flowline import FileModel
from oggm import cfg, utils
from oggm.core import gis

# Module logger
log = logging.getLogger(__name__)



def surf_to_nan(surf_h, thick):

    t1 = thick[:-2]
    t2 = thick[1:-1]
    t3 = thick[2:]
    pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
    surf_h[np.where(pnan)[0] + 1] = np.nan
    return surf_h




def plot_modeloutput_section_New(model=None, ax=None, title=''):
    """Plots the result of the model output along the flowline.

    Parameters
    ----------
    model: obj
        either a FlowlineModel or a list of model flowlines.
    fig
    title
    """

    try:
        fls = model.fls
    except AttributeError:
        fls = model

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    else:
        fig = plt.gcf()

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in fls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0,  posax.width / 3.0,
             posax.height]
    axh = fig.add_axes(posax, frameon=False)

    axh.hist(height, orientation='horizontal', range=ylim, bins=20,
             alpha=0.3, weights=area)
    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area incl. tributaries (km$^2$)')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    cls = fls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed')

    # Where trapezoid change color
    # if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
    #     bed_t = cls.bed_h * np.nan
    #     pt = cls.is_trapezoid & (~cls.is_rectangular)
    #     bed_t[pt] = cls.bed_h[pt]
    #     ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
    #             label='Bed (Trap.)')
    #     bed_t = cls.bed_h * np.nan
    #     bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
    #     ax.plot(x, bed_t, color='crimson', linewidth=2.5, label='Bed (Rect.)')

    # Plot glacier
    surfh = surf_to_nan(cls.surface_h, cls.thick)
    ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

    # Plot tributaries
    for i, l in zip(cls.inflow_indices, cls.inflows):
        if l.thick[-1] > 0:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='#993399',
                    markeredgecolor='k',
                    label='Tributary (active)')
        else:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='w',
                    markeredgecolor='k',
                    label='Tributary (inactive)')
    if getattr(model, 'do_calving', False):
        ax.hlines(model.water_level, x[0], x[-1], linestyles=':', color='C0')

    ax.set_ylim(ylim)
    # add the horizontal line, y=0
    ax.axhline(y=0,linestyle='--',color='k',linewidth=0.5)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Title
    ax.set_title(title, loc='left')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(1.34, 1.0),
              frameon=False)
    # plt.show()
    # plt.close()S