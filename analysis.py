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
    """PLot of wavelen data and coefficients approximation.

    Args:
        df ([DataFrame]): [description].
        waveletname (str): ['coif5', 'sym4', 'sym5'].

    Returns:
        [tuple[array, array]]: wavelen data and coefficients.
    """
    t = df.Seconds.to_numpy() / (15*60)
    data = df.Load.to_numpy().copy()

    plt.plot(t, data, c='k')
    plt.show()

    levels = pywt.dwt_max_level(len(data), waveletname)
    data_l, c_l = pywt.dwt(data, waveletname)
    dec = 2.
    fig, axarr = plt.subplots(nrows=levels, ncols=2, figsize=(20, 20))
    for i in range(levels):
        axarr[i, 0].plot(data_l, 'r')
        axarr[i, 0].set_xlim(t[0] // dec, t[-1] // dec)
        axarr[i, 1].plot(c_l, 'g')
        axarr[i, 1].set_xlim(t[0] // dec, t[-1] // dec)
        axarr[i, 0].set_ylabel(f"Level {i + 1}", fontsize=14, rotation=90)
        #axarr[i, 0].set_yticklabels([])
        dec *= 2.
        if i == 0:
            axarr[i, 0].set_title("Approximation coefficients", fontsize=14)        
            axarr[i, 1].set_title("Detail coefficients", fontsize=14)        
        #axarr[i, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()
    return data_l, c_l
