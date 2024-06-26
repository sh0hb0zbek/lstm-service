def plotting(data, method, freq='A', interpolation=None, aspect='auto',
             start_row=None, end_row=None, title=None, lags=50, ax=None, line='r'):
    """
        'L'------> Line Plot
        'D.'-----> Dot Plot
        'H'------> Histogram
        'De'-----> Density Plot
        'S'------> Scatter Plot
        'A'------> Autocorrelation Plot
        'ACF'----> Autocorrelation Function Plot
        'PACF'---> Partial Autocorrelation Function Plot
        'QQ'-----> Q-Q Plot
        'SL'-----> Stacked Line Plot
        'B'------> Box Plot
        'HM'-----> Heat Map
    """
    from matplotlib import pyplot
    from pandas import Grouper
    from pandas import DataFrame
    if method in ['L', 'Line']:
        if isinstance(data, list):
            for i in range(len(data)):
                plt = DataFrame(data[i])
                pyplot.plot(plt)
        else:
            data.plot(ax=ax)
    if method in ['D.', 'Dot']:
        data.plot(style='k.')
    if method in ['H', 'Histogram']:
        data.hist(ax=ax)
    if method in ['De', 'Density']:
        data.plot(kind='kde', ax=ax)
    if method in ['S', 'Scatter']:
        from pandas.plotting import lag_plot
        lag_plot(data, ax=ax)
    if method in ['A', 'Autocorrelation']:
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(data, ax=ax)
    if method in ['ACF']:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(data, lags=lags, ax=ax)
    if method in ['PACF']:
        from statsmodels.graphics.tsaplots import plot_pacf
        plot_pacf(data, lags=lags, ax=ax)
    if method in ['QQ']:
        from statsmodels.graphics.gofplots import qqplot
        qqplot(data, line=line, ax=ax)
    if method in ['SL', 'StackedLine',
                  'B', 'Box',
                  'HM', 'HeatMap']:
        groups = data[start_row:end_row].groupby(Grouper(freq=freq))
        years = DataFrame()
        for name, group in groups:
            years[name.year] = group.values
        if method in ['SL', 'StackedLine']:
            years.plot(subplots=True, legend=False)
        if method in ['B', 'Box']:
            years.boxplot()
        if method in ['HM', 'HeatMap']:
            years = years.T
            pyplot.matshow(years, interpolation=interpolation, aspect=aspect)
    if title is not None:
        pyplot.title(title)
    pyplot.show()


def evaluate_error(test, pred, metrics='RMSE'):
    """
      'MAE'  --> Mean Absolute Error
      'MSE'  --> Mean Squared Error
      'RMSE' --> Root Mean Squared Error
    """
    if metrics in ['MeanAbsoluteError', 'MAE']:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(test, pred)
    elif metrics in ['MeanSquaredError', 'MSE']:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(test, pred)
    elif metrics in ['all', 'RootMeanSquaredError', 'RMSE']:
        from math import sqrt
        return sqrt(evaluate_error(test=test, pred=pred, metrics='MSE'))
    elif metrics == 'accuracy_score':
        from sklearn.metrics import accuracy_score
        return accuracy_score(test, pred)
    elif metrics == 'confusion_matrix':
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(test, pred)
