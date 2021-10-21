import numpy as np 

def normalize(data, data_max, data_min):
    return (data - data_min) / (data_max - data_min)

def denormalize(data_normalized, data_max, data_min):
    return data_normalized * (data_max - data_min) + data_min

def q1(x):
    return x.quantile(0.025)

def q2(x):
    return x.quantile(0.975)