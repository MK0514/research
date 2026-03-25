"""
ISA (International Standard Atmosphere) Model
ISO 2533:1975 Standard

Reference:
  - ISO 2533:1975, "Standard Atmosphere"
  - U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF)
  - Anderson, J.D. (2007), "Introduction to Flight", McGraw-Hill, 6th ed., pp. 64-80

Author: Launch Vehicle Aero Suite
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AtmosphericState:
    """대기 상태량 (Atmospheric state variables)"""
    altitude: float     # m (고도)
    temperature: float  # K (온도)
    pressure: float     # Pa (압력)
    density: float      # kg/m³ (밀도)
    speed_of_sound: float  # m/s (음속)
    dynamic_viscosity: float  # Pa·s (동점성계수)
    kinematic_viscosity: float  # m²/s (동적점성계수)
    mach_number: float = 0.0  # 마하수 (주어진 속도에서)


class ISAModel:
    """
    International Standard Atmosphere (ISA) Model
    
    층 구조 (Layer Structure):
      - Troposphere:   0 ~ 11,000 m  (lapse rate: -6.5 K/km)
      - Tropopause:  11,000 ~ 20,000 m  (isothermal: 216.65 K)
      - Stratosphere 1: 20,000 ~ 32,000 m (lapse rate: +1.0 K/km)
      - Stratosphere 2: 32,000 ~ 47,000 m (lapse rate: +2.8 K/km)
      - Stratopause:  47,000 ~ 51,000 m  (isothermal: 270.65 K)
      - Mesosphere 1: 51,000 ~ 71,000 m  (lapse rate: -2.8 K/km)
      - Mesosphere 2: 71,000 ~ 86,000 m  (lapse rate: -2.0 K/km)

    References:
      - ISO 2533:1975
      - ICAO Doc 7488/3 (1993)
    """

    # ISA 상수 (Constants)
    R_AIR   = 287.058    # J/(kg·K) 공기 기체상수 (specific gas constant for air)
    GAMMA   = 1.4        # 비열비 (ratio of specific heats)
    G0      = 9.80665    # m/s² 표준중력가속도 (standard gravity)
    T_SL    = 288.15     # K 해수면 온도 (sea-level temperature)
    P_SL    = 101325.0   # Pa 해수면 압력 (sea-level pressure)
    RHO_SL  = 1.225      # kg/m³ 해수면 밀도 (sea-level density)
    A_SL    = 340.294    # m/s 해수면 음속 (sea-level speed of sound)

    # 층 경계 고도 (Layer boundary altitudes, m)
    # [고도, 기준온도, 온도 기울기(K/m)]
    LAYERS = [
        (0,      288.15, -0.0065),   # Troposphere
        (11000,  216.65,  0.0000),   # Tropopause (isothermal)
        (20000,  216.65, +0.0010),   # Stratosphere 1
        (32000,  228.65, +0.0028),   # Stratosphere 2
        (47000,  270.65,  0.0000),   # Stratopause (isothermal)
        (51000,  270.65, -0.0028),   # Mesosphere 1
        (71000,  214.65, -0.0020),   # Mesosphere 2
        (86000,  186.87,  0.0000),   # Upper mesosphere (approx)
    ]

    def __init__(self, delta_T: float = 0.0):
        """
        delta_T: ISA deviation (hot/cold day offset, K)
        """
        self.delta_T = delta_T

    def get_state(self, altitude_m: float, velocity_ms: float = 0.0) -> AtmosphericState:
        """
        고도와 속도에 대한 대기 상태량 계산
        Calculate atmospheric state at given altitude and velocity

        Args:
            altitude_m  : 기하학적 고도 (geometric altitude) [m]
            velocity_ms : 비행 속도 [m/s]

        Returns:
            AtmosphericState dataclass
        """
        altitude_m = np.clip(altitude_m, 0, 86000)
        T, P = self._temperature_pressure(altitude_m)
        T += self.delta_T  # ISA deviation 적용

        rho = P / (self.R_AIR * T)
        a   = np.sqrt(self.GAMMA * self.R_AIR * T)
        mu  = self._sutherland_viscosity(T)
        nu  = mu / rho
        M   = velocity_ms / a if a > 0 else 0.0

        return AtmosphericState(
            altitude=altitude_m,
            temperature=T,
            pressure=P,
            density=rho,
            speed_of_sound=a,
            dynamic_viscosity=mu,
            kinematic_viscosity=nu,
            mach_number=M,
        )

    def _temperature_pressure(self, h: float) -> Tuple[float, float]:
        """
        층별 온도-압력 계산 (Layer-by-layer T and P)
        Uses hydrostatic equation + ideal gas law.
        
        Reference: U.S. Standard Atmosphere (1976), Eq. 23a, 23b
        """
        T = self.LAYERS[0][1]
        P = self.P_SL

        for i in range(len(self.LAYERS) - 1):
            h_base, T_base, L = self.LAYERS[i]
            h_top = self.LAYERS[i + 1][0]

            if h <= h_top:
                dh = h - h_base
                if abs(L) < 1e-10:   # Isothermal layer (등온층)
                    P = P * np.exp(-self.G0 * dh / (self.R_AIR * T_base))
                else:                 # Gradient layer (온도기울기층)
                    T = T_base + L * dh
                    P = P * (T / T_base) ** (-self.G0 / (L * self.R_AIR))
                return T, P
            else:
                # 층 경계까지 전파 (Propagate to layer boundary)
                dh = h_top - h_base
                T_top = self.LAYERS[i + 1][1]
                if abs(L) < 1e-10:
                    P = P * np.exp(-self.G0 * dh / (self.R_AIR * T_base))
                else:
                    P = P * (T_top / T_base) ** (-self.G0 / (L * self.R_AIR))
                T = T_top

        return T, P

    def _sutherland_viscosity(self, T: float) -> float:
        """
        Sutherland의 점성계수 공식
        
        Reference:
          - Sutherland, W. (1893). "The viscosity of gases and molecular force."
            Philosophical Magazine, 36, 507–531.
          - White, F.M. (2006). Viscous Fluid Flow, 3rd ed., p.28 (McGraw-Hill)

        mu = mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
        S = 110.4 K (Sutherland constant for air)
        """
        T_ref  = 291.15   # K
        mu_ref = 1.827e-5  # Pa·s
        S      = 120.0    # K (Sutherland constant)
        return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

    def get_dynamic_pressure(self, altitude_m: float, velocity_ms: float) -> float:
        """동압 계산 q = 0.5 * rho * V^2 [Pa]"""
        state = self.get_state(altitude_m, velocity_ms)
        return 0.5 * state.density * velocity_ms ** 2

    def get_trajectory_states(self, altitudes: np.ndarray,
                               velocities: np.ndarray) -> list:
        """궤도 시뮬레이션용 일괄 계산"""
        return [self.get_state(h, v) for h, v in zip(altitudes, velocities)]


if __name__ == "__main__":
    isa = ISAModel()
    print("=== ISA 표준 대기 검증 (ISA Standard Atmosphere Verification) ===")
    print(f"{'고도(m)':>10} {'온도(K)':>10} {'압력(Pa)':>12} {'밀도(kg/m³)':>12} {'음속(m/s)':>10}")
    print("-" * 56)
    for h in [0, 5000, 11000, 20000, 32000, 47000, 71000]:
        st = isa.get_state(h)
        print(f"{h:>10.0f} {st.temperature:>10.2f} {st.pressure:>12.2f} "
              f"{st.density:>12.5f} {st.speed_of_sound:>10.2f}")
