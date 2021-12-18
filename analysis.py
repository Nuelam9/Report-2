import numpy as np 
import matplotlib.pyplot as plt
import pywt

def normalize(data, data_max, data_min):
    return (data - data_min) / (data_max - data_min)

def denormalize(data_normalized, data_max, data_min):
    return data_normalized * (data_max - data_min) + data_min

def q1(x):
    return x.quantile(0.025)

def q2(x):
    return x.quantile(0.975)

def wavelet_denoising(df, waveletname='sym4'):
    """[summary]

    Args:
        df ([DataFrame]): [description].
        waveletname (str): ['coif5', 'sym4', 'sym5'].
    """
    t = df.Seconds.to_numpy() / (15*60)
    data = df.Load.to_numpy().copy()

    plt.plot(t, data, c='k')
    plt.show()

    levels = pywt.dwt_max_level(len(data), waveletname)
    dec=2
    fig, axarr = plt.subplots(nrows=levels, ncols=2, figsize=(20,20))
    for i in range(levels):
        (globals()[f'data_l{i + 1}d'],
         globals()[f'c_l{i + 1}d']) = pywt.dwt(data, waveletname)
        axarr[i, 0].plot(globals()[f'data_l{i + 1}d'], 'r')
        axarr[i, 0].set_xlim(t[0] // dec, t[-1] // dec)
        axarr[i, 1].plot(globals()[f'c_l{i + 1}d'], 'g')
        axarr[i, 1].set_xlim(t[0] // dec, t[-1] // dec)
        axarr[i, 0].set_ylabel(f"Level {i + 1}", fontsize=14, rotation=90)
        #axarr[i, 0].set_yticklabels([])
        dec *= 2
        if i == 0:
            axarr[i, 0].set_title("Approximation coefficients", fontsize=14)        
            axarr[i, 1].set_title("Detail coefficients", fontsize=14)        
        #axarr[i, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()