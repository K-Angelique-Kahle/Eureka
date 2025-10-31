import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class StepModel(Model):
    """Model for step-functions in time"""
    def __init__(self, **kwargs):
        """Initialize the step-function model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'step'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Per-channel suffix for param key lookup: _ch# and optional _wl#.
        self._suffix_by_chan = {}
        for chan, wl in zip(self.fitted_channels, self.wl_groups):
            sfx = ''
            if chan > 0:
                sfx += f'_ch{chan}'
            if wl > 0:
                sfx += f'_wl{wl}'
            self._suffix_by_chan[chan] = sfx

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
                self.time_local[trim1:trim2] = piece - piece.data[0]
        else:
            self.time_local = self._time - self._time.data[0]

    def _index_set_for_chan(self, chan):
        """Discover step indices for a given channel.

        Scans ``self.parameters.dict`` for keys that match the suffix for
        this channel.

        Accepted patterns per index N are:
          * ``step{N}{sfx}`` and ``steptime{N}{sfx}``

        Only indices present in *both* sets are returned.

        Parameters
        ----------
        chan : int
            Real channel id.

        Returns
        -------
        list of int
            Sorted indices ``N`` present for both step and steptime.
        """
        if getattr(self, "parameters", None) is None:
            return []

        keys = getattr(self.parameters, "dict", {}).keys()
        sfx = self._suffix_by_chan[chan]

        def parse_idx(key, prefix, sfx_):
            """Return N if key == f'{prefix}{{N}}{sfx_}', else None.

            If ``sfx_`` is empty (chan==0 & wl==0), accept unsuffixed
            keys; otherwise enforce the exact non-empty suffix.
            """
            if not key.startswith(prefix):
                return None
            if sfx_ and not key.endswith(sfx_):
                return None
            end = len(key) - len(sfx_) if sfx_ else len(key)
            mid = key[len(prefix):end]
            return int(mid) if mid.isdigit() else None

        step_idx = set()
        time_idx = set()

        for k in keys:
            i = parse_idx(k, "step", sfx)
            if i is not None:
                step_idx.add(i)
            i = parse_idx(k, "steptime", sfx)
            if i is not None:
                time_idx.add(i)

        # Only accept indices that have both a step and a steptime.
        return sorted(step_idx.intersection(time_idx))

    def _read_steps_for_chan(self, chan):
        """Read and sort step pairs for a given channel.

        For each index ``N`` discovered by ``_index_set_for_chan``,
        read values via ``_get_param_value`` using the same key rules
        as in ``_match_and_index``. Pairs with zero amplitude are
        skipped. The result is sorted by step time.

        Parameters
        ----------
        chan : int
            Real channel id.

        Returns
        -------
        list of tuple
            A list of ``(t_step, step)`` pairs sorted by ``t_step``.
        """
        idxs = self._index_set_for_chan(chan)
        sfx = self._suffix_by_chan[chan]
        pairs = []

        for n in idxs:
            # Build exact keys using the per-channel suffix.
            k_step = f"step{n}{sfx}"
            k_time = f"steptime{n}{sfx}"
            step = self._get_param_value(k_step)
            tstep = self._get_param_value(k_time)

            if step == 0.0:
                continue
            pairs.append((tstep, step))

        # Ensure deterministic application order.
        pairs.sort(key=lambda x: x[0])
        return pairs

    def eval(self, channel=None, **kwargs):
        """Evaluate the step model.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one channel. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The model values at self.time.
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
                # Split the arrays that have lengths of the original time axis
                t = split([t], self.nints, chan)[0]

            lcpiece = np.ma.ones(t.shape)
            for tstep, step in self._read_steps_for_chan(chan):
                mask = t >= tstep
                lcpiece[mask] = lcpiece[mask] + step

            lcpiece = np.ma.masked_where(np.ma.getmaskarray(t), lcpiece)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
