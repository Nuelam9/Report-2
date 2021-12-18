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


def wavelet_denoising(df, waveletname):
    t = df.Seconds[:5096].values / (15*60)
    data = df.Load[:5096].copy()

    plt.plot(t, data, c='k')
    plt.show()

    #waveletname = 'sym5' #  'sym4' #'coif5'  #

    levels=10
    dec=2
    fig, axarr = plt.subplots(nrows=levels, ncols=2, figsize=(20,20))
    for ii in range(levels):
        (globals()['data_l%d' %(ii + 1)], globals()['c_l%d' %(ii + 1)]) = pywt.dwt(data, waveletname)
        axarr[ii, 0].plot(globals()['data_l%d' %(ii + 1)], 'r')
        axarr[ii, 0].set_xlim(t[0]//dec,t[-1]//dec)
        axarr[ii, 1].plot(globals()['c_l%d' %(ii + 1)], 'g')
        axarr[ii, 1].set_xlim(t[0]//dec,t[-1]//dec)
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        #axarr[ii, 0].set_yticklabels([])
        dec=dec*2
        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)        
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)        
        #axarr[ii, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()