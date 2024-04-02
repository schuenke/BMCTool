import math

import numpy as np

from bmctool.parameters import Parameters


class BlochMcConnellSolver:
    """Solver class for Bloch-McConnell equations."""

    def __init__(self, params: Parameters, n_offsets: int) -> None:
        """Init method for Bloch-McConnell solver.

        Parameters
        ----------
        params
            Parameters object
        n_offsets
            number of offsets
        """
        self.params: Parameters = params
        self.n_offsets: int = n_offsets
        self.n_pools: int = params.num_cest_pools
        self.size: int = params.m_vec.size
        self.arr_a: np.ndarray
        self.arr_c: np.ndarray
        self.w0: float
        self.dw0: float

        self.update_params(params)

    def _init_matrix_a(self) -> None:
        """Initialize self.arr_a with all parameters from self.params."""
        n_p = self.n_pools

        # Create a 2D array with dimensions (size, size)
        self.arr_a = np.zeros((self.size, self.size), dtype=float)

        # Set mt_pool parameters
        k_ac = 0.0
        if self.params.mt_pool is not None:
            k_ca = self.params.mt_pool.k
            k_ac = k_ca * self.params.mt_pool.f
            self.arr_a[2 * (n_p + 1), 3 * (n_p + 1)] = k_ca
            self.arr_a[3 * (n_p + 1), 2 * (n_p + 1)] = k_ac

        # Set water_pool parameters
        k1a = self.params.water_pool.r1 + k_ac
        k2a = self.params.water_pool.r2
        for pool in self.params.cest_pools:
            k_ai = pool.f * pool.k
            k1a += k_ai
            k2a += k_ai

        self.arr_a[0, 0] = -k2a
        self.arr_a[1 + n_p, 1 + n_p] = -k2a
        self.arr_a[2 + 2 * n_p, 2 + 2 * n_p] = -k1a

        # Set cest_pools parameters
        for i, pool in enumerate(self.params.cest_pools):
            k_ia = pool.k
            k_ai = k_ia * pool.f
            k_1i = k_ia + pool.r1
            k_2i = k_ia + pool.r2

            self.arr_a[0, i + 1] = k_ia
            self.arr_a[i + 1, 0] = k_ai
            self.arr_a[i + 1, i + 1] = -k_2i

            self.arr_a[1 + n_p, i + 2 + n_p] = k_ia
            self.arr_a[i + 2 + n_p, 1 + n_p] = k_ai
            self.arr_a[i + 2 + n_p, i + 2 + n_p] = -k_2i

            self.arr_a[2 * (n_p + 1), i + 1 + 2 * (n_p + 1)] = k_ia
            self.arr_a[i + 1 + 2 * (n_p + 1), 2 * (n_p + 1)] = k_ai
            self.arr_a[i + 1 + 2 * (n_p + 1), i + 1 + 2 * (n_p + 1)] = -k_1i

    def _init_vector_c(self) -> None:
        """Initialize vector self.C with all parameters from self.params."""
        n_p = self.n_pools

        # initialize vector c with dimensions (size, 1)
        self.arr_c = np.zeros((self.size, 1), dtype=float)

        # Set water_pool parameters
        self.arr_c[(n_p + 1) * 2, 0] = self.params.water_pool.f * self.params.water_pool.r1

        # Set parameters for all cest pools
        self.arr_c[(n_p + 1) * 2 + 1 : (n_p + 1) * 2 + 1 + len(self.params.cest_pools), 0] = [
            pool.f * pool.r1 for pool in self.params.cest_pools
        ]

        # Set mt_pool parameters
        if self.params.mt_pool is not None:
            self.arr_c[3 * (n_p + 1), 0] = self.params.mt_pool.f * self.params.mt_pool.r1

    def update_params(self, params: Parameters) -> None:
        """Update matrix self.A according to given Parameters object.

        Parameters
        ----------
        params
            Parameters object
        """
        self.params = params
        self.w0 = params.system.b0 * params.system.gamma
        self.dw0 = self.w0 * params.system.b0_inhom
        self._init_matrix_a()
        self._init_vector_c()

    def update_matrix(self, rf_amp: float, rf_phase: float, rf_freq: float) -> None:
        """Update matrix self.A according to given parameters.

        Parameters
        ----------
        rf_amp
            rf amplitude [Hz]
        rf_phase
            rf phase [rad]
        rf_freq
            rf frequency [Hz]
        """
        n_p: int = self.n_pools

        # set dw0 due to b0_inhomogeneity
        self.arr_a[0, 1 + n_p] = -self.dw0
        self.arr_a[1 + n_p, 0] = self.dw0

        # calculate omega_1
        rf_amp_2pi = rf_amp * 2 * np.pi * self.params.system.rel_b1
        rf_amp_2pi_sin = rf_amp_2pi * np.sin(rf_phase)
        rf_amp_2pi_cos = rf_amp_2pi * np.cos(rf_phase)

        # set omega_1 for water_pool
        self.arr_a[0, 2 * (n_p + 1)] = -rf_amp_2pi_sin
        self.arr_a[2 * (n_p + 1), 0] = rf_amp_2pi_sin

        self.arr_a[n_p + 1, 2 * (n_p + 1)] = rf_amp_2pi_cos
        self.arr_a[2 * (n_p + 1), n_p + 1] = -rf_amp_2pi_cos

        # set omega_1 for cest pools
        i_values = np.arange(n_p + 1)
        self.arr_a[i_values, i_values + 2 * (n_p + 1)] = -rf_amp_2pi_sin
        self.arr_a[i_values + 2 * (n_p + 1), i_values] = rf_amp_2pi_sin

        self.arr_a[n_p + 1 + i_values, i_values + 2 * (n_p + 1)] = rf_amp_2pi_cos
        self.arr_a[i_values + 2 * (n_p + 1), n_p + 1 + i_values] = -rf_amp_2pi_cos

        # set off-resonance terms for water pool
        rf_freq_2pi = rf_freq * 2 * np.pi
        self.arr_a[0, 1 + n_p] += rf_freq_2pi
        self.arr_a[1 + n_p, 0] -= rf_freq_2pi

        # set off-resonance terms for cest pools
        dwi_values = np.array([pool.dw for pool in self.params.cest_pools]) * self.w0 - (rf_freq_2pi + self.dw0)
        indices = np.arange(1, n_p + 1)
        self.arr_a[indices, indices + n_p + 1] = -dwi_values
        self.arr_a[indices + n_p + 1, indices] = dwi_values

        # mt_pool
        if self.params.mt_pool is not None:
            self.arr_a[3 * (n_p + 1), 3 * (n_p + 1)] = (
                -self.params.mt_pool.r1
                - self.params.mt_pool.k
                - rf_amp_2pi**2 * self.get_mt_shape_at_offset(rf_freq_2pi + self.dw0, self.w0)
            )

    def solve_equation(self, mag: np.ndarray, dtp: float) -> np.ndarray:
        """Solve one step of BMC equations using the Padé approximation.

        This function is not used atm.
        :param mag: magnetization vector before current step
        :param dtp: duration of current step
        :return: magnetization vector after current step
        """
        arr_a = np.squeeze(self.arr_a)
        arr_c = np.squeeze(self.arr_c)
        mag_ = np.squeeze(mag)
        n_iter = 6  # number of iterations
        a_inv_t = np.dot(np.linalg.pinv(arr_a), arr_c)
        a_t = np.dot(arr_a, dtp)

        _, inf_exp = math.frexp(np.linalg.norm(a_t, ord=np.inf))
        j = max(0, inf_exp)
        a_t = a_t * (1 / pow(2, j))

        x = a_t.copy()
        c = 0.5
        n = np.identity(arr_a.shape[0])
        d = n - c * a_t
        n = n + c * a_t

        p = True
        for k in range(2, n_iter + 1):
            c = c * (n_iter - k + 1) / (k * (2 * n_iter - k + 1))
            x = np.dot(a_t, x)
            c_x = c * x
            n = n + c_x
            d = d + c_x if p else d - c_x
            p = not p

        f = np.dot(np.linalg.pinv(d), n)
        for _ in range(1, j + 1):
            f = np.dot(f, f)
        mag_ = np.dot(f, (mag_ + a_inv_t)) - a_inv_t
        return mag_[:, np.newaxis]

    def get_mt_shape_at_offset(self, offset: float, w0: float) -> float:
        """Calculate the lineshape of the MT pool at the given offset(s).

        :param offsets: frequency offset(s)
        :param w0: Larmor frequency of simulated system
        :return: lineshape of mt pool at given offset(s)
        """
        if not self.params.mt_pool:
            return 0

        ls = self.params.mt_pool.lineshape.lower()
        dw = self.params.mt_pool.dw
        t2 = 1 / self.params.mt_pool.r2
        if ls == 'lorentzian':
            mt_line = t2 / (1 + pow((offset - dw * w0) * t2, 2.0))
        elif ls == 'superlorentzian':
            dw_pool = offset - dw * w0
            mt_line = self.interpolate_sl(dw_pool) if abs(dw_pool) >= w0 else self.interpolate_chs(dw_pool, w0)
        else:
            mt_line = 0.0
        return mt_line  # type: ignore

    def interpolate_sl(self, dw: float) -> float:
        """Interpolate MT profile for SuperLorentzian lineshape.

        :param dw: relative frequency offset
        :return: MT profile at given relative frequency offset
        """
        if not self.params.mt_pool:
            return 0

        mt_line = 0
        t2 = 1 / self.params.mt_pool.r2
        n_samples = 101
        step_size = 0.01
        sqrt_2pi = np.sqrt(2 / np.pi)
        for i in range(n_samples):
            powcu2 = abs(3 * pow(step_size * i, 2) - 1)
            mt_line += sqrt_2pi * t2 / powcu2 * np.exp(-2 * pow(dw * t2 / powcu2, 2))
        return mt_line * np.pi * step_size

    def interpolate_chs(self, dw_pool: float, w0: float) -> float:
        """Cubic Hermite Spline Interpolation."""
        mt_line = 0
        px = np.array([-300 - w0, -100 - w0, 100 + w0, 300 + w0])
        py = np.zeros(px.size)
        for i in range(px.size):
            py[i] = self.interpolate_sl(px[i])
        if px.size != 4 or py.size != 4:
            return mt_line
        else:
            tan_weight = 30
            d0y = tan_weight * (py[1] - py[0])
            d1y = tan_weight * (py[3] - py[2])
            c_step = abs((dw_pool - px[1] + 1) / (px[2] - px[1] + 1))
            h0 = 2 * (c_step**3) - 3 * (c_step**2) + 1
            h1 = -2 * (c_step**3) + 3 * (c_step**2)
            h2 = (c_step**3) - 2 * (c_step**2) + c_step
            h3 = (c_step**3) - (c_step**2)

            mt_line = h0 * py[1] + h1 * py[2] + h2 * d0y + h3 * d1y
            return mt_line
