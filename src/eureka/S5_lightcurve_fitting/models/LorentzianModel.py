import numpy as np

from .Model import Model
from ...lib.split_channels import split


class LorentzianModel(Model):
    """An asymmetric Lorentzian model"""

    def __init__(self, **kwargs):
        """Initialize the asymmetric Lorentzian model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'lorentzian'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Build per-channel suffix for param key lookup (_ch#/_wl#).
        self._suffix_by_chan = {}
        for chan, wl in zip(self.fitted_channels, self.wl_groups):
            sfx = ''
            if chan > 0:
                sfx += f'_ch{chan}'
            if wl > 0:
                sfx += f'_wl{wl}'
            self._suffix_by_chan[chan] = sfx

    def _get_optional(self, key):
        """Return float value for param key or None if missing/invalid.

        Based on _get_param_value, but allows for missing parameters.

        Parameters
        ----------
        key : str
            Parameter key to look up.
        """
        if getattr(self, "parameters", None) is None:
            return None

        val = getattr(self.parameters, key, None)
        if val is None:
            return None

        if hasattr(val, "value"):
            val = val.value

        try:
            out = float(val)
        except (TypeError, ValueError, OverflowError):
            return None

        return out

    def _read_params_for_chan(self, chan):
        """Read Lorentzian params for a channel.

        Parameters
        ----------
        chan : int
            Channel index.

        Returns
        -------
        tuple
            Parameters tuple with possible None values.
        """
        sfx = self._suffix_by_chan.get(chan, '')

        params = (self._get_optional(f'lor_amp{sfx}'),
                  self._get_optional(f'lor_amp_lhs{sfx}'),
                  self._get_optional(f'lor_amp_rhs{sfx}'),
                  self._get_optional(f'lor_hwhm{sfx}'),
                  self._get_optional(f'lor_hwhm_lhs{sfx}'),
                  self._get_optional(f'lor_hwhm_rhs{sfx}'),
                  self._get_param_value(f'lor_t0{sfx}', 0.),
                  self._get_param_value(f'lor_power{sfx}', 2.))

        return params

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

            t = self.time
            if self.multwhite:
                t = split([t], self.nints, chan)[0]

            amp, amp_lhs, amp_rhs, hwhm, hwhm_lhs, hwhm_rhs, t0, power = \
                self._read_params_for_chan(chan)

            # Branch selection:
            # A) Symmetric: amp & hwhm set; no side-specific params.
            is_sym = (
                amp is not None and hwhm is not None and
                amp_lhs is None and amp_rhs is None and
                hwhm_lhs is None and hwhm_rhs is None
            )
            # B) Asym widths only: amp set; hwhm_lhs & hwhm_rhs set.
            is_asym_width = (
                amp is not None and hwhm is None and
                hwhm_lhs is not None and hwhm_rhs is not None and
                amp_lhs is None and amp_rhs is None
            )
            # C) Asym amps + widths: all side-specific set; no global amp.
            is_asym_amp_width = (
                amp is None and hwhm is None and
                amp_lhs is not None and amp_rhs is not None and
                hwhm_lhs is not None and hwhm_rhs is not None
            )

            if is_asym_amp_width:
                lhs = np.ma.where(t <= t0)
                rhs = np.ma.where(t > t0)
                ut = np.ma.zeros(t.shape)
                ut[lhs] = (t0 - t[lhs]) / hwhm_lhs
                ut[rhs] = (t[rhs] - t0) / hwhm_rhs

                piece = np.ma.zeros(t.shape)
                baseline = 1. + amp_lhs - amp_rhs
                piece[lhs] = 1. + amp_lhs / (1. + ut[lhs]**power)
                piece[rhs] = baseline + amp_rhs / (1. + ut[rhs]**power)
            elif is_asym_width:
                lhs = np.ma.where(t <= t0)
                rhs = np.ma.where(t > t0)
                ut = np.ma.zeros(t.shape)
                ut[lhs] = (t[lhs] - t0) / hwhm_lhs
                ut[rhs] = (t[rhs] - t0) / hwhm_rhs

                piece = 1. + amp / (1. + ut**power)
            elif is_sym:
                ut = 2. * (t - t0) / hwhm
                piece = 1. + amp / (1. + ut**power)
            else:
                raise ValueError(
                    "Ambiguous Lorentzian parameterization. Use one of:\n"
                    "  1) lor_amp, lor_hwhm\n"
                    "  2) lor_amp, lor_hwhm_lhs, lor_hwhm_rhs\n"
                    "  3) lor_amp_lhs, lor_amp_rhs, lor_hwhm_lhs, lor_hwhm_rhs"
                )

            pieces.append(piece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
