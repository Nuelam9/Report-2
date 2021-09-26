from analysis import * 


def _psf_prediction(dataset, n_forecast, seasonality):
    '''
    dataset: pandas.series
             Data to perform prediction
    
    n_forecast: int
                Number of values to predict 
    
    seasonality: int
                 Seasonality of the time series
    '''

    n_forecast_seasons = int(n_forecast / seasonality)
    n = 1

    f = {'Load': ['mean', q1, q2]}
    pred = dataset.groupby(['Day', 'Minutes', 'Hour']).agg(f)
    