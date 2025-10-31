import numpy as np

from .Model import Model
from ...lib.split_channels import split


class DampedOscillatorModel(Model):
    """A damped oscillator model"""
    def __init__(self, **kwargs):
        """Initialize the damped oscillator model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        super().__init__(**kwargs)
        self.name = 'damped oscillator'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Build suffix per real channel id for param key lookup.
        # Supports optional _ch# and _wl# suffixes.
        self._suffix_by_chan = {}
        for chan, wl in zip(self.fitted_channels, self.wl_groups):
            suffix = ''
            if chan > 0:
                suffix += f'_ch{chan}'
            if wl > 0:
                suffix += f'_wl{wl}'
            self._suffix_by_chan[chan] = suffix

    def _read_params_for_chan(self, chan):
        """Return oscillator params for the given channel.

        Returns
        -------
        tuple
            (amp0, amp_decay, per0, per_decay, t0, t1)
        """
        sfx = self._suffix_by_chan.get(chan)

        amp0 = self._get_param_value(f'osc_amp{sfx}')
        amp_decay = self._get_param_value(f'osc_amp_decay{sfx}')
        per0 = self._get_param_value(f'osc_per{sfx}')
        per_decay = self._get_param_value(f'osc_per_decay{sfx}')
        t0 = self._get_param_value(f'osc_t0{sfx}')
        t1 = self._get_param_value(f'osc_t1{sfx}')

        return amp0, amp_decay, per0, per_decay, t0, t1

    def eval(self, channel=None, **kwargs):
        """Evaluate the model at the current (or provided) times.

        Parameters
        ----------
        channel : int; optional
            If not None, evaluate only this channel. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The model value at self.time.
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for i in range(nchan):
            chan_id = channels[i] if self.nchannel_fitted > 1 else 0

            t = self.time
            if self.multwhite:
                t = split([t], self.nints, chan_id)[0]

            (amp0, amp_decay, per0, per_decay, t0, t1) = \
                self._read_params_for_chan(chan_id)

            amp = amp0 * np.exp(-amp_decay * (t - t0))
            per = per0 * np.exp(-per_decay * (t - t0))
            osc = 1. + amp * np.sin(2 * np.pi * (t - t1) / per)
            # Force pre-t0 region to unity.
            osc[t < t0] = 1.

            pieces.append(osc)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
