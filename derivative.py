

class Derivative:
    """
    The idea of this class is that it is completely independent from the model
    or dcf assumptions. It is an vessel to stroke information (maturity, etc)
    and the properties of subclasses are the interface that the pricer functions
    read when pricing.

    calibrate will return the market price, also for calibration purposes
    """

    def __init__(self, market_price=None, maturity=None, principal=None, **kwargs):

        self._maturity = maturity
        self._name = 'Derivative'

        if principal is None:
            self._principal = 1

        else:
            self._principal = principal

        self._market_price = market_price

    def __repr__(self):

        if self._market_price is None:
            line = '<{} with maturity {}>'
            return line.format(self._name, self.maturity)

        else:
            line = '<{} with maturity {} and price of {:6.5f}>'
            return line.format(self._name,
                               self.maturity,
                               self._market_price)

    @property
    def maturity(self):
        return self._maturity

    @property
    def principal(self):
        return self._principal

    @property
    def calibrate(self):
        return self._market_price

    @calibrate.setter
    def calibrate(self, x):
        self._market_price = x



class ZeroCouponBond(Derivative):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Zero coupon bond'


class OptionOnZeroCouponBond(Derivative):

    def __init__(self, strike_date=None, strike_price=None, **kwargs):

        super().__init__(**kwargs)

        self._strike_date = strike_date
        self._strike_price = strike_price

    def __repr__(self):

        if self._market_price is None:
            line = '<{} of maturity {}, with strike of {} at {}>'
            return line.format(self._name,
                               self.maturity,
                               self.strike_date,
                               self.strike_price)

        else:
            line = '<{} of maturity {}, with strike of {} at {} and market price of {:6.5f}>'
            return line.format(self._name,
                               self.maturity,
                               self.strike_date,
                               self.strike_price,
                               self._market_price)

    @property
    def strike_date(self):
        return self._strike_date

    @property
    def strike_price(self):
        return self._strike_price

    @property
    def underlying_bond(self):
        return ZeroCouponBond(maturity=self.maturity,
                              principal=self.principal)

    @property
    def bond_to_strike(self):
        return ZeroCouponBond(maturity=self.strike_date,
                              principal=self.principal)

    @property
    def call(self):
        return CallOnZeroCouponBond(maturity=self.maturity,
                                    principal=self.principal,
                                    strike_date=self.strike_date,
                                    strike_price=self.strike_price)


class CallOnZeroCouponBond(OptionOnZeroCouponBond):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Call option on zero coupon bond'


class PutOnZeroCouponBond(OptionOnZeroCouponBond):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Put option on zero coupon bond'


class CapletOrFloorlet(Derivative):

    def __init__(self, strike_date=None, strike_rate=None, **kwargs):

        super().__init__(**kwargs)

        self._strike_date = strike_date
        self._strike_rate = strike_rate

    def __repr__(self):

        if self._market_price is None:
            line = '<{} with strike of {} at {}>'
            return line.format(self._name,
                               self.strike_date,
                               self.strike_rate)

        else:
            line = '<{} with strike of {} at {} and market price of {:6.5f}>'
            return line.format(self._name,
                               self.strike_date,
                               self.strike_rate,
                               self._market_price)

    @property
    def strike_date(self):
        return self._strike_date

    @property
    def strike_rate(self):
        return self._strike_rate


class Caplet(CapletOrFloorlet):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Caplet'


class Floorlet(CapletOrFloorlet):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Floorlet'


class CapOrFloor(Derivative):

    def __init__(self, coupon_dates=None, strike_rate=None, **kwargs):

        super().__init__(**kwargs)

        self._coupon_dates = coupon_dates
        self._strike_rate = strike_rate

    @property
    def coupon_dates(self):
        return self._coupon_dates

    @property
    def strike_rate(self):
        return self._strike_rate


class Cap(CapOrFloor):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Cap'


class Floor(CapOrFloor):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._name = 'Floor'


class CDS(Derivative):

    pass

