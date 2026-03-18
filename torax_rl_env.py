"""Tokamak plasma-control environment with optional real TORAX integration.

The environment supports two backends:
- a real TORAX transport solve with time-varying actuator schedules; and
- a lightweight mock backend for fast development or unit tests.

The TORAX path is intentionally more expensive but materially more serious than
an ad-hoc mock: each control step rebuilds a time-dependent TORAX config using
all actions taken so far, then reruns the transport simulation over the full
control horizon. This is slower than a true in-memory closed-loop coupling, but
it ensures that observations come from the real transport solver instead of from
placeholder algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import torax
    from torax import StateHistory, run_simulation

    HAS_TORAX = True
except ImportError:  # pragma: no cover - optional dependency
    torax = None
    StateHistory = Any
    run_simulation = None
    HAS_TORAX = False

try:
    from fusion_surrogates.qlknn import qlknn_model  # noqa: F401

    HAS_QLKNN = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_QLKNN = False


logger = logging.getLogger(__name__)


@dataclass
class ToraxEnvironmentConfig:
    """Configuration for the tokamak plasma-control environment."""

    # Simulation timing
    t_final: float = 1.0
    fixed_dt: float = 0.1
    max_dt: float = 0.2
    episode_max_steps: int = 8

    # Backend selection
    use_torax: bool = True
    use_qlknn: bool = True
    use_mock_fallback: bool = True

    # Machine geometry
    R_major: float = 6.2
    a_minor: float = 2.0
    B_0: float = 5.3
    elongation_lcfs: float = 1.72
    n_rho: int = 15

    # Initial conditions
    T_e_core_init: float = 8.0
    T_i_core_init: float = 8.0
    n_e_core_init: float = 1.1e20
    n_e_edge: float = 0.6e20
    I_p_init: float = 8.0e6
    Z_eff: float = 1.6

    # Actuators
    P_heating_max: float = 50.0e6
    P_heating_baseline: float = 18.0e6
    particle_source_baseline: float = 1.0e21
    generic_current_fraction: float = 0.12
    current_ramp_rate_limit: float = 1.0e6  # A/s
    heating_location: float = 0.25
    heating_width: float = 0.12
    current_location: float = 0.36
    current_width: float = 0.075
    particle_width: float = 0.25
    particle_deposition_location: float = 0.30

    # Operating limits/targets
    I_p_min: float = 5.0e6
    I_p_max: float = 12.0e6
    n_Greenwald_frac: float = 0.9
    beta_N_limit: float = 3.0
    target_T_e: float = 6.0
    target_n_e: float = 0.95  # in 1e20 m^-3 for observations
    target_q95: float = 4.5
    target_beta_N: float = 1.8
    target_tau_E: float = 2.5

    # TORAX numerics / solver
    adaptive_dt: bool = False
    use_pereverzev: bool = True
    theta_implicit: float = 1.0

    # Constant-transport fallback params when QLKNN is disabled
    chi_i: float = 1.2
    chi_e: float = 1.0
    D_e: float = 0.2
    V_e: float = -0.05

    # Mock backend development knobs
    scenario_randomization: bool = True
    actuator_lag: float = 0.35
    temperature_relaxation: float = 0.08
    density_relaxation: float = 0.04
    radiation_loss_coeff: float = 0.018
    transport_loss_coeff: float = 0.55
    bootstrap_coupling: float = 0.05
    obs_normalize: bool = False
    act_normalize: bool = True

    # Internal schedule seed values for TORAX rollout construction
    control_history_seed: Tuple[float, ...] = field(default_factory=lambda: (0.0,))


class ToraxRLEnvironment(gym.Env):
    """Gymnasium environment for tokamak plasma control."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config: Optional[ToraxEnvironmentConfig] = None,
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or ToraxEnvironmentConfig()
        self.render_mode = render_mode
        self.verbose = verbose

        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        if self.config.use_torax and not HAS_TORAX:
            logger.warning("TORAX not available, falling back to mock plasma backend")
            self.config.use_torax = False
        if self.config.use_qlknn and not HAS_QLKNN:
            logger.warning("QLKNN not available, falling back to constant transport in TORAX")
            self.config.use_qlknn = False

        self.current_state_history: Optional[StateHistory] = None
        self.step_count = 0
        self.episode_count = 0
        self._prev_obs = np.zeros(11, dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

        # Mock backend state
        self._state: Dict[str, float] = {}

        # TORAX control schedule state
        self._control_times: list[float] = []
        self._heating_schedule: list[float] = []
        self._ip_schedule: list[float] = []
        self._torax_runtime_backend: str = "mock"
        self._torax_sim_error: str = "NONE"

        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 11, dtype=np.float32),
            high=np.array([np.inf] * 11, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        self._last_action = np.zeros(2, dtype=np.float32)
        self._torax_sim_error = "NONE"

        if self.config.use_torax:
            try:
                obs = self._reset_torax_backend()
                self._torax_runtime_backend = "torax"
            except Exception as exc:
                if not self.config.use_mock_fallback:
                    raise
                logger.warning("TORAX reset failed (%s); using mock backend", exc)
                self.config.use_torax = False
                obs = self._reset_mock_backend()
                self._torax_runtime_backend = "mock"
        else:
            obs = self._reset_mock_backend()
            self._torax_runtime_backend = "mock"

        self._prev_obs = obs.copy()
        return obs, {
            "episode": self.episode_count,
            "backend": self._torax_runtime_backend,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._last_action = action.copy()

        if self.config.use_torax:
            try:
                obs = self._step_torax_backend(action)
                self._torax_runtime_backend = "torax"
            except Exception as exc:
                if not self.config.use_mock_fallback:
                    raise
                logger.warning("TORAX step failed (%s); using mock backend", exc)
                self.config.use_torax = False
                if not self._state:
                    self._reset_mock_backend()
                obs = self._step_mock_backend(action)
                self._torax_runtime_backend = "mock"
        else:
            obs = self._step_mock_backend(action)
            self._torax_runtime_backend = "mock"

        reward = self._compute_reward(obs, action)
        terminated = self._check_termination(obs)
        truncated = self.step_count >= self.config.episode_max_steps
        info = {
            "step": self.step_count,
            "backend": self._torax_runtime_backend,
            "P_heating": float(action[0] * self.config.P_heating_max / 1.0e6),
            "reward_components": self._get_reward_components(obs, action),
            "torax_sim_error": self._torax_sim_error,
        }
        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Real TORAX backend
    # ---------------------------------------------------------------------
    def _reset_torax_backend(self) -> np.ndarray:
        self._control_times = [0.0, self.config.fixed_dt]
        self._heating_schedule = [self.config.P_heating_baseline, self.config.P_heating_baseline]
        self._ip_schedule = [self.config.I_p_init, self.config.I_p_init]
        state_history = self._run_torax_schedule(final_time=self.config.fixed_dt)
        return self._extract_observation_from_torax(state_history)

    def _step_torax_backend(self, action: np.ndarray) -> np.ndarray:
        current_time = self._control_times[-1]
        next_time = current_time + self.config.fixed_dt

        heating_power = float(np.clip(action[0], 0.0, 1.0) * self.config.P_heating_max)
        ramp_command = float(np.clip(action[1], -2.0, 2.0) * self.config.current_ramp_rate_limit)
        prev_ip = self._ip_schedule[-1]
        next_ip = float(np.clip(prev_ip + ramp_command * self.config.fixed_dt, self.config.I_p_min, self.config.I_p_max))

        self._control_times.append(next_time)
        self._heating_schedule.append(heating_power)
        self._ip_schedule.append(next_ip)

        state_history = self._run_torax_schedule(final_time=next_time)
        return self._extract_observation_from_torax(state_history)

    def _run_torax_schedule(self, final_time: float) -> StateHistory:
        torax_config = self._build_torax_config(final_time)
        _, state_history = run_simulation(
            torax_config,
            log_timestep_info=False,
            progress_bar=False,
        )
        self.current_state_history = state_history
        self._torax_sim_error = str(getattr(state_history, "sim_error", "NONE"))
        return state_history

    def _build_torax_config(self, final_time: float) -> Any:
        control_times = self._control_times
        heat_schedule = {t: p for t, p in zip(control_times, self._heating_schedule)}
        ip_schedule = {t: p for t, p in zip(control_times, self._ip_schedule)}

        transport_config: Dict[str, Any]
        if self.config.use_qlknn:
            transport_config = {"model_name": "qlknn"}
        else:
            transport_config = {
                "model_name": "constant",
                "chi_i": self.config.chi_i,
                "chi_e": self.config.chi_e,
                "D_e": self.config.D_e,
                "V_e": self.config.V_e,
            }

        config_dict = {
            "profile_conditions": {
                "Ip": ip_schedule,
                "T_i": {0.0: {0.0: self.config.T_i_core_init, 1.0: 1.0}},
                "T_i_right_bc": 1.0,
                "T_e": {0.0: {0.0: self.config.T_e_core_init, 1.0: 1.0}},
                "T_e_right_bc": 1.0,
                "n_e": {0.0: {0.0: self.config.n_e_core_init, 1.0: self.config.n_e_edge}},
                "n_e_right_bc": self.config.n_e_edge,
                "n_e_right_bc_is_fGW": False,
                "normalize_n_e_to_nbar": False,
                "nbar": self.config.n_e_core_init,
                "n_e_nbar_is_fGW": False,
            },
            "numerics": {
                "t_initial": 0.0,
                "t_final": final_time,
                "fixed_dt": self.config.fixed_dt,
                "max_dt": self.config.max_dt,
                "adaptive_dt": self.config.adaptive_dt,
                "evolve_ion_heat": True,
                "evolve_electron_heat": True,
                "evolve_current": True,
                "evolve_density": True,
            },
            "plasma_composition": {
                "main_ion": {"D": 0.5, "T": 0.5},
                "impurity": "Ne",
                "Z_eff": self.config.Z_eff,
            },
            "geometry": {
                "geometry_type": "circular",
                "R_major": self.config.R_major,
                "a_minor": self.config.a_minor,
                "B_0": self.config.B_0,
                "elongation_LCFS": self.config.elongation_lcfs,
                "n_rho": self.config.n_rho,
            },
            "sources": {
                "generic_heat": {
                    "P_total": heat_schedule,
                    "electron_heat_fraction": 0.8,
                    "gaussian_location": self.config.heating_location,
                    "gaussian_width": self.config.heating_width,
                },
                "generic_current": {
                    "fraction_of_total_current": self.config.generic_current_fraction,
                    "gaussian_width": self.config.current_width,
                    "gaussian_location": self.config.current_location,
                },
                "generic_particle": {
                    "S_total": self.config.particle_source_baseline,
                    "particle_width": self.config.particle_width,
                    "deposition_location": self.config.particle_deposition_location,
                },
            },
            "transport": transport_config,
            "pedestal": {
                "model_name": "no_pedestal",
                "set_pedestal": False,
            },
            "solver": {
                "solver_type": "linear",
                "theta_implicit": self.config.theta_implicit,
                "use_pereverzev": self.config.use_pereverzev,
            },
        }
        return torax.ToraxConfig(**config_dict)

    def _extract_observation_from_torax(self, state_history: StateHistory) -> np.ndarray:
        cp = state_history.core_profiles[-1]
        pp = state_history.post_processed_outputs[-1]

        te = float(np.asarray(pp.T_e_volume_avg))
        ti = float(np.asarray(pp.T_i_volume_avg))
        ne = float(np.asarray(pp.n_e_volume_avg) / 1.0e20)
        ip = float(np.asarray(cp.Ip_profile_face[-1]) / 1.0e6)
        q95 = float(np.asarray(pp.q95))
        beta_n = float(np.asarray(pp.beta_N))
        power_loss_mw = float(np.asarray(pp.P_SOL_total) / 1.0e6)
        tau_e = float(np.asarray(pp.W_thermal_total) / max(np.asarray(pp.P_SOL_total), 1.0))

        dt = max(self.config.fixed_dt, 1e-6)
        if np.any(self._prev_obs):
            dte_dt = (te - float(self._prev_obs[0])) / dt
            dne_dt = (ne - float(self._prev_obs[2])) / dt
            dip_dt = (ip - float(self._prev_obs[3])) / dt
        else:
            dte_dt = 0.0
            dne_dt = 0.0
            dip_dt = 0.0

        return np.array(
            [te, ti, ne, ip, q95, beta_n, tau_e, dte_dt, dne_dt, dip_dt, power_loss_mw],
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # Mock backend retained for fast dev/tests
    # ---------------------------------------------------------------------
    def _reset_mock_backend(self) -> np.ndarray:
        rng = self.np_random
        spread = 0.0 if not self.config.scenario_randomization else 1.0
        self._state = {
            "Te": float(self.config.T_e_core_init + spread * rng.uniform(-0.8, 0.8)),
            "Ti": float(self.config.T_i_core_init + spread * rng.uniform(-0.7, 0.7)),
            "ne": float(self.config.target_n_e + spread * rng.uniform(-0.15, 0.15)),
            "Ip": float(self.config.I_p_init / 1.0e6 + spread * rng.uniform(-0.4, 0.4)),
            "dTe_dt": 0.0,
            "dne_dt": 0.0,
            "dIp_dt": 0.0,
            "P_loss": 18.0,
            "tau_E": 0.55,
        }
        return self._build_mock_observation()

    def _step_mock_backend(self, action: np.ndarray) -> np.ndarray:
        dt = self.config.fixed_dt
        heating_frac = float(action[0])
        current_ramp_target = float(action[1])

        te = self._state["Te"]
        ti = self._state["Ti"]
        ne = self._state["ne"]
        ip = self._state["Ip"]
        d_ip_prev = self._state["dIp_dt"]

        heating_mw = heating_frac * self.config.P_heating_max / 1.0e6
        current_ramp = d_ip_prev + self.config.actuator_lag * (current_ramp_target - d_ip_prev)
        ip = np.clip(ip + current_ramp * dt, self.config.I_p_min / 1.0e6, self.config.I_p_max / 1.0e6)

        tau_e = self._compute_tau_E_mock(heating_mw, ip, ne)
        pressure_index = ne * (te + ti)
        beta_n = self._compute_beta_N_mock(pressure_index, ip)
        q95 = self._compute_q95_mock(ip)
        n_greenwald = self._greenwald_density_mock(ip)

        dte_dt = (
            0.09 * heating_mw
            + 0.18 * tau_e
            - self.config.radiation_loss_coeff * ne * te**1.35
            - self.config.transport_loss_coeff * max(te - 1.0, 0.0) / max(tau_e, 0.2)
            - max(0.0, 2.4 - q95) * 0.7
            - max(0.0, beta_n - self.config.target_beta_N) * 0.45
            - self.config.temperature_relaxation * (te - self.config.target_T_e)
        )
        dti_dt = 0.8 * dte_dt + self.config.bootstrap_coupling * current_ramp
        dne_dt = (
            0.012 * heating_mw
            + 0.02 * max(0.0, 0.75 - ne / max(n_greenwald, 1e-6))
            - self.config.density_relaxation * (ne - self.config.target_n_e)
            - 0.015 * max(0.0, ne - 0.82 * n_greenwald)
        )

        noise_scale = 0.015 if self.config.scenario_randomization else 0.0
        rng = self.np_random
        te = np.clip(te + dt * dte_dt + rng.normal(0.0, noise_scale), 0.5, 20.0)
        ti = np.clip(ti + dt * dti_dt + rng.normal(0.0, noise_scale), 0.5, 20.0)
        ne = np.clip(ne + dt * dne_dt + rng.normal(0.0, noise_scale * 0.3), 0.2, 4.0)
        tau_e = self._compute_tau_E_mock(heating_mw, ip, ne)
        power_loss = self._compute_power_loss_mock(ne, te, ti, tau_e)

        self._state.update(
            {
                "Te": float(te),
                "Ti": float(ti),
                "ne": float(ne),
                "Ip": float(ip),
                "dTe_dt": float(dte_dt),
                "dne_dt": float(dne_dt),
                "dIp_dt": float(current_ramp),
                "P_loss": float(power_loss),
                "tau_E": float(tau_e),
                "beta_N": float(self._compute_beta_N_mock(ne * (te + ti), ip)),
                "q95": float(self._compute_q95_mock(ip)),
            }
        )
        return self._build_mock_observation()

    def _build_mock_observation(self) -> np.ndarray:
        return np.array(
            [
                self._state["Te"],
                self._state["Ti"],
                self._state["ne"],
                self._state["Ip"],
                self._state.get("q95", self._compute_q95_mock(self._state["Ip"])),
                self._state.get("beta_N", self._compute_beta_N_mock(self._state["ne"] * (self._state["Te"] + self._state["Ti"]), self._state["Ip"])),
                self._state["tau_E"],
                self._state["dTe_dt"],
                self._state["dne_dt"],
                self._state["dIp_dt"],
                self._state["P_loss"],
            ],
            dtype=np.float32,
        )

    def _compute_tau_E_mock(self, heating_mw: float, ip_ma: float, ne_20: float) -> float:
        return float(np.clip(0.12 * (ip_ma ** 0.85) * (self.config.B_0 ** 0.3) * (max(ne_20, 0.2) ** 0.1) / (max(heating_mw + 5.0, 1.0) ** 0.35), 0.15, 2.5))

    def _compute_q95_mock(self, ip_ma: float) -> float:
        return float(np.clip(4.5 * (self.config.I_p_init / 1.0e6) / max(ip_ma, 1e-6), 1.3, 8.0))

    def _compute_beta_N_mock(self, pressure_index: float, ip_ma: float) -> float:
        return float(np.clip(0.12 * pressure_index / max(self.config.B_0, 0.1) * ((self.config.I_p_init / 1.0e6) / max(ip_ma, 1e-6)) ** 0.25, 0.05, 8.0))

    def _greenwald_density_mock(self, ip_ma: float) -> float:
        return float(ip_ma / (np.pi * self.config.a_minor**2))

    def _compute_power_loss_mock(self, ne_20: float, te_keV: float, ti_keV: float, tau_E: float) -> float:
        return float(np.clip(ne_20 * (te_keV + ti_keV) * 3.2 / max(tau_E, 0.2), 1.0, 80.0))

    # ---------------------------------------------------------------------
    # Shared reward / termination logic
    # ---------------------------------------------------------------------
    def _tracking_score(self, value: float, target: float, scale: float) -> float:
        return float(np.exp(-abs(value - target) / max(scale, 1e-6)))

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        te, _, ne, _, q95, beta_n, tau_e, _, _, dIp, _ = obs
        reward = (
            0.24 * self._tracking_score(float(te), self.config.target_T_e, 1.5)
            + 0.18 * self._tracking_score(float(ne), self.config.target_n_e, 0.18)
            + 0.20 * self._tracking_score(float(q95), self.config.target_q95, 0.8)
            + 0.16 * self._tracking_score(float(beta_n), self.config.target_beta_N, 0.5)
            + 0.14 * min(float(tau_e) / self.config.target_tau_E, 1.0)
            + 0.05 * (1.0 / (1.0 + abs(float(dIp))))
            + 0.03 * (1.0 - 0.15 * float(np.clip(action[0], 0.0, 1.0)))
        )
        return float(np.clip(reward, 0.0, 1.0))

    def _get_reward_components(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        te, _, ne, _, q95, beta_n, tau_e, _, _, dIp, _ = obs
        return {
            "temperature_tracking": self._tracking_score(float(te), self.config.target_T_e, 1.5),
            "density_tracking": self._tracking_score(float(ne), self.config.target_n_e, 0.18),
            "q95_tracking": self._tracking_score(float(q95), self.config.target_q95, 0.8),
            "beta_tracking": self._tracking_score(float(beta_n), self.config.target_beta_N, 0.5),
            "confinement": min(float(tau_e) / self.config.target_tau_E, 1.0),
            "current_smoothness": 1.0 / (1.0 + abs(float(dIp))),
            "heating_efficiency": 1.0 - 0.15 * float(np.clip(action[0], 0.0, 1.0)),
        }

    def _check_termination(self, obs: np.ndarray) -> bool:
        te, _, ne, ip, q95, beta_n, _, _, _, _, _ = obs
        n_GW = self._greenwald_density_mock(float(ip))
        torax_error = self._torax_runtime_backend == "torax" and self._torax_sim_error not in {"SimError.NO_ERROR", "NONE"}
        return bool(
            te < 1.0
            or q95 < 2.0
            or ne > self.config.n_Greenwald_frac * n_GW
            or beta_n > 1.3 * self.config.beta_N_limit
            or torax_error
        )

    def render(self) -> None:
        if self.render_mode == "human":
            logger.info("Step %s | backend=%s | obs=%s", self.step_count, self._torax_runtime_backend, self._prev_obs)

    def close(self) -> None:
        logger.info("Environment closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = ToraxRLEnvironment(ToraxEnvironmentConfig(use_torax=True, use_qlknn=False, episode_max_steps=2), verbose=True)
    obs, info = env.reset(seed=0)
    print("Initial observation:", obs)
    for step in range(2):
        action = np.array([0.4, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}, backend={info['backend']}")
        if terminated or truncated:
            break
