import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class ExpRampModel(Model):
    """Model for single or double exponential ramps"""
    def __init__(self, **kwargs):
        """Initialize the exponential ramp model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'exp. ramp'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Define the coefficient keys per channel
        # Here we assume two exponentials: (r0, r1) and (r2, r3).
        self.r_keys_per_chan = {}
        for chan, wl in zip(self.fitted_channels, self.wl_groups):
            suffix = ''
            if chan > 0:
                suffix += f'_ch{chan}'
            if wl > 0:
                suffix += f'_wl{wl}'
            self.r_keys_per_chan[chan] = [f'r0{suffix}', f'r1{suffix}',
                                          f'r2{suffix}', f'r3{suffix}']

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        if time_array is None:
            self._time = None
            self.time_local = None
            return

        self._time = np.ma.masked_invalid(time_array)
        # Convert to local time
        if self.multwhite:
            self.time_local = np.ma.zeros(self._time.shape)
            for chan in self.fitted_channels:
                # Split the arrays that have lengths
                # of the original time axis
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                # Use .data[0] to be robust to masks
                self.time_local[trim1:trim2] = piece - piece.data[0]
        else:
            # Use .data[0] to be robust to masks
            self.time_local = self._time - self._time.data[0]

    def _read_coeff_tuple_for_chan(self, chan):
        """Read (r0, r1, r2, r3) for the requested channel."""
        keys = self.r_keys_per_chan[chan]
        return tuple(self._get_param_value(k, 0.) for k in keys)

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The ramp model evaluated at self.time (or provided time).
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the ramp from the coeffs
        pieces = []
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            t = self.time_local
            if self.multwhite:
                # Split arrays that have lengths of the original time axis
                t = split([t], self.nints, chan)[0]

            r0, r1, r2, r3 = self._read_coeff_tuple_for_chan(chan)
            lcpiece = 1. + r0*np.exp(-r1*t) + r2*np.exp(-r3*t)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
