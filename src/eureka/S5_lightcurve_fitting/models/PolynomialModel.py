import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class PolynomialModel(Model):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'polynomial'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Build per-channel coefficient keys keyed by real channel id.
        # Coeff names: c0..c9 (+ optional _ch#/_wl# suffixes).
        self.c_keys_per_chan = {}
        for chan, wl in zip(self.fitted_channels, self.wl_groups):
            suffix = ''
            if chan > 0:
                suffix += f'_ch{chan}'
            if wl > 0:
                suffix += f'_wl{wl}'
            self.c_keys_per_chan[chan] = [f'c{i}{suffix}' for i in range(10)]

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
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                self.time_local[trim1:trim2] = piece - piece.mean()
        else:
            self.time_local = self._time - self._time.mean()

    def _read_coeffs_desc_for_chan(self, chan):
        """Return poly coeffs in descending order for a given channel.

        We read c0..c9, trim trailing zeros, then return
        [cN, cN-1, ..., c0] suitable for np.polyval.
        If all zeros, return [0.0].
        """
        keys = self.c_keys_per_chan[chan]  # c0-c9
        vals = np.array([self._get_param_value(k) for k in keys])

        # Trim high-degree trailing zeros.
        nonzero = np.nonzero(vals)[0]
        if nonzero.size == 0:
            trimmed = np.array([0.], dtype=float)
        else:
            max_idx = int(nonzero[-1])
            trimmed = vals[:max_idx+1]

        # Descending order for np.polyval
        return trimmed[::-1]

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
            The value of the model at self.time.
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            t = self.time_local
            if self.multwhite:
                t = split([t], self.nints, chan)[0]

            coeffs_desc = self._read_coeffs_desc_for_chan(chan)
            lcpiece = np.polyval(coeffs_desc, t)
            lcpiece = np.ma.masked_where(np.ma.getmaskarray(t), lcpiece)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
