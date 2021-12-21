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

def fourierExtrapolation(x, n_predict, n_harm=10):
    n = x.size

    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)

    x_notrend = x - p[0] * t - p[1]     # signal detrended 
    x_freqdom = np.fft.fft(x_notrend)   # signal in frequencies domain
    f = np.fft.fftfreq(n)               # frequencies

    indexes = list(range(n))
    indexes.sort(key= lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sign = np.zeros(t.size)

    for i in indexes[:1 + n_harm * 2]:
        amplitude = np.absolute(x_freqdom[i]) / n
        phase = np.angle(x_freqdom[i])
        restored_sign += amplitude * np.cos(2 * np.pi * f[i] * t + phase)
    
    return restored_sign + p[0] * t + p[1]

def wavelet_denoising(df, waveletname='sym4'):
    """Plot of wavelen data and coefficients approximation.

    Args:
        df ([DataFrame]): [description].
        waveletname (str): ['coif5', 'sym4', 'sym5'].

    Returns:
        [tuple[array, array]]: wavelen data and coefficients.
    """
    t = np.arange(len(df))
    data = df.Load.to_numpy().copy()
    
    plt.plot(data, c='k', lw=0.1)
    plt.show()

    levels = pywt.dwt_max_level(len(data), waveletname)
    data_l, c_l = pywt.dwt(data, waveletname)
    lws = np.array([0.1 if i < 3 else 0.5 for i in range(levels)])

    # Empirical evidence suggests that a good initial guess for the 
    # decomposition depth is about half of the maximum possible depth
    dec = 2.
    fig, axs = plt.subplots(nrows=levels, ncols=2, figsize=(20, 20))
    for i in range(levels):
        axs[i, 0].plot(data_l, 'r', lw=lws[i])
        axs[i, 0].set_xlim(t[0] // dec, t[-1] // dec)
        axs[i, 1].plot(c_l, 'g', lw=lws[i])
        axs[i, 1].set_xlim(t[0] // dec, t[-1] // dec)
        axs[i, 0].set_ylabel(f"Level {i + 1}", fontsize=14, rotation=90)
        dec *= 2.
        if i == 0:
            axs[i, 0].set_title("Approximation coefficients", fontsize=14)        
            axs[i, 1].set_title("Detail coefficients", fontsize=14)        
    plt.tight_layout()
    plt.show()
    return data_l, c_l
