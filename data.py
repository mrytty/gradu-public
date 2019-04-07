import pandas as pd
import datetime

from curve import LinearCurve, QuadraticCurve, CubicCurve, CubicSplineCurve


def date_handler(suspect_date):
    """
    Modifies objects that should be datetime.date objects
    :param suspect_date: an object that should be datetime.date object
    :return: datetime.date object functionally equivalent to given param
    """

    # Nothing to see here
    if isinstance(suspect_date, datetime.date):
        return suspect_date

    # Strip time part
    elif isinstance(suspect_date, datetime.datetime):
        return suspect_date.date()

    # Read using datetime.datetime.strptime and modify to date. Assume '%Y-%m-%d' form.
    elif isinstance(suspect_date, str):
        return datetime.datetime.strptime(suspect_date, '%Y-%m-%d').date()

    # Laziness
    else:
        raise NotImplementedError


class DataContainer:

    def __init__(self, today=None, *args, **kwargs):

        self._type = 'general'
        self._data = {}
        self._name = 'Data'
        self._txt = ''

        if today is not None:
            self._today = date_handler(today)

    def __repr__(self):

        ret = 'Contains {} data. Available dates are:'.format(self._type)
        for key in self._data:
            ret += '\n' + str(key)

        return ret

    @property
    def txt(self):
        return self._txt

    @txt.setter
    def txt(self, x):
        if isinstance(x, str):
            self._txt = x

    @property
    def today(self):
        return self._today

    @property
    def data(self):
        return self._data

    @property
    def series(self):
        return pd.Series(self._data, name=self._name)

    @property
    def df(self):
        series = self.series
        return pd.DataFrame(series)

    @data.setter
    def data(self, x):

        # If given dict, just update (overrides existing keys
        if isinstance(x, dict):
            self._data.update(x)

        # if tuple, we assume that the first point is date and the second point is numerical value
        elif isinstance(x, tuple):
            date, point, *rest, = x
            date = date_handler(date)

            # Notifies if updated
            if date in self._data:
                print('Date {} is updated'.format(date))

            self._data[date] = point

        # If given a list, use recursion
        elif isinstance(x, list):
            for y in x:
                self.data = y

    def get(self, handle):
        """
        Returns data for given date/dates
        :param handle: date/dates
        :return: data point / list of data
        """

        # If it is a list, use recursion
        if isinstance(handle, list):
            return [self.get(x) for x in handle]

        # Use date handler and return data
        else:
            date = date_handler(handle)
            return self.data[date]

    def date_list_handler(self, suspect_date_list):

        return [date_handler(suspect_date) for suspect_date in suspect_date_list]


class RateContainer(DataContainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Overwrite parent values
        self._type = 'rate'
        self._name = 'Rates'

    def curve(self, today=None, curve='CubicSpline', dcf='Act360'):

        # Choose curve class
        if curve == 'CubicSpline':
            curve_type = CubicSplineCurve

        elif curve == 'Cubic':
            curve_type = CubicCurve

        elif curve == 'QuadraticCurve':
            curve_type = QuadraticCurve

        else:
            curve_type = LinearCurve

        # Date checking
        if today is None:
            today = self._today

        today = date_handler(today)

        # Check if the date is before data dates
        if today < min(self.series.index):

            # Create
            curve = curve_type(today=today, dcf=dcf)

            # Input data
            curve.give_rates(self.series/100)

            # Output the object
            return curve

        else:
            raise ValueError

    def zeros(self, dates, by='Rates', curve='CubicSpline', dcf='Act360', method='Act360'):

        curve = self.curve(curve=curve, dcf=dcf)
        return curve.generate_zeros(dates, by=by, method=method)

    def plot(self, save=None):

        title = '{}: {}'.format(self.txt, self._today)
        self.curve().plot(title=title, save=save)






ios_data = [('2018-07-27', -0.366), ('2018-08-06', -0.3558), ('2018-08-30', -0.3632), ('2018-10-30', -0.3619), ('2019-01-30', -0.3615), ('2019-07-30', -0.3577), ('2021-07-30', -0.1549), ('2023-07-31', 0.1219), ('2026-07-30', 0.4841), ('2028-07-31', 0.7062), ('2033-07-29', 1.0862), ('2038-07-30', 1.2774), ('2048-07-30', 1.3653)]
swap_data = [('2018-07-27', -0.43), ('2018-08-06', -0.4094), ('2018-08-30', -0.3728), ('2018-10-30', -0.3065), ('2019-01-30', -0.2757), ('2019-07-30', -0.2573), ('2021-07-30', -0.01), ('2023-07-31', 0.3058), ('2026-07-30', 0.7073), ('2028-07-31', 0.9302), ('2033-07-29', 1.3060), ('2038-07-30', 1.4841), ('2048-07-30', 1.5502)]

germany_data = [('2019-01-30', -0.6), ('2021-07-30', -0.486), ('2023-07-31', -0.226), ('2026-07-30', 0.163), ('2028-07-31', 0.405), ('2038-07-30', 0.771), ('2048-07-30', 1.059)]
france_data = [('2019-01-30', -0.544), ('2021-07-30', -0.394), ('2023-07-31', 0.054), ('2026-07-30', 0.43), ('2028-07-31', 0.705), ('2033-07-29', 1.088), ('2038-07-30', 1.219), ('2048-07-30', 1.581)]
italy_data = [('2019-01-30', 0.217), ('2021-07-30', 1.04), ('2026-07-30', 2.4), ('2028-07-31', 2.705), ('2033-07-29', 3.072), ('2038-07-30', 3.33), ('2048-07-30', 3.497)]

date_string = '2018-07-26'

ios = RateContainer(date_string)
swap = RateContainer(date_string)
germany = RateContainer(date_string)
france = RateContainer(date_string)
italy = RateContainer(date_string)

today = ios.today

ios.data = ios_data
swap.data = swap_data
germany.data = germany_data
france.data = france_data
italy.data = italy_data

ios.txt = 'Overnight index swap curve'
swap.txt = 'Swap curve'
germany.txt = 'Germany rate curve'
france.txt = 'France rate curve'
italy.txt = 'Italy rate curve'

test_dates = ios.series.index[:-1]


