import matplotlib as mpl
import numpy as np 

def latex_settings():

    fig_width_pt = 222.62206                # Get this from LaTeX using \the\columnwidth

    inches_per_pt = 1.0/72.27               # Convert pt to inches

    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio

    fig_width = fig_width_pt*inches_per_pt  # width in inches

    fig_height =fig_width*golden_mean       # height in inches

    fig_size = [fig_width,fig_height]

    params = {'backend': 'ps',

            'axes.labelsize': 10,

            'legend.fontsize': 10,

            'xtick.labelsize': 8,

            'ytick.labelsize': 8,

            'figure.figsize': fig_size}

    return params