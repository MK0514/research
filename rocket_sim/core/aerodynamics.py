"""
공기역학 해석 엔진 (Aerodynamic Analysis Engine)
발사 단계별 공기역학 특성 계산 (Phase-by-phase aerodynamic analysis)

Flight Phases:
  1. 발사 준비 (Pre-launch)         : M = 0
  2. 아음속 (Subsonic)             : 0 < M < 0.8
  3. MaxQ (Maximum Dynamic Pressure): M ≈ 1.0–1.3 (최대 동압)
  4. 천음속 (Transonic)            : 0.8 ≤ M ≤ 1.2
  5. 초음속 (Supersonic)           : 1.2 < M ≤ 5.0
  6. 극초음속 (Hypersonic)         : M > 5.0

References:
  - Anderson, J.D. (2003). Modern Compressible Flow, 3rd ed. McGraw-Hill.
  - Hoerner, S.F. (1965). Fluid-Dynamic Drag. Published by the author.
  - USAF DATCOM (1978). USAF Stability and Control Datcom. Wright-Patterson AFB.
  - NASA TM-2011-216962 (2011). "Rocket Aerodynamics." NASA.
  - Lees, L. (1956). "Laminar heat transfer over blunt-nosed bodies at
    hypersonic flight speeds." Jet Propulsion, 26(4), 259–269.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum, auto

from core.atmosphere import ISAModel, AtmosphericState
from core.shock_wave import (oblique_shock, normal_shock,
                              prandtl_meyer_expansion,
                              wave_drag_coefficient, friction_drag_coefficient,
                              mach_angle, bow_shock_standoff)
from geometry.shapes import VehicleGeometry, NoseType


class FlightPhase(Enum):
    PRE_LAUNCH  = "발사 준비 (Pre-Launch)"
    SUBSONIC    = "아음속 (Subsonic)"
    TRANSONIC   = "천음속 (Transonic)"
    MAX_Q       = "최대 동압 (Max-Q)"
    SUPERSONIC  = "초음속 (Supersonic)"
    HYPERSONIC  = "극초음속 (Hypersonic)"


@dataclass
class ShockFeature:
    """충격파/팽창파 특성 (Shock/Expansion feature)"""
    feature_type: str       # 'oblique_shock', 'normal_shock', 'expansion_fan'
    x_location: float       # 발사체 좌표 [m]
    angle: float            # 충격각 or 팽창각 [deg]
    M_upstream: float
    M_downstream: float
    pressure_ratio: float
    temperature_ratio: float
    label: str = ""


@dataclass
class AeroAnalysisResult:
    """공기역학 해석 결과 (Aerodynamic analysis result)"""
    mach: float
    altitude: float         # m
    velocity: float         # m/s
    phase: FlightPhase

    # 대기 상태
    atm: AtmosphericState

    # 항력 성분 [무차원]
    Cd_wave: float          # 파동 항력계수
    Cd_friction: float      # 마찰 항력계수
    Cd_base: float          # 기저 항력계수
    Cd_total: float         # 전체 항력계수

    # 항력 [N]
    dynamic_pressure: float # q [Pa]
    drag_force: float       # D [N]

    # 충격파/팽창파 특성 목록
    shock_features: List[ShockFeature] = field(default_factory=list)

    # 표면 압력 분포 (Surface pressure distribution)
    x_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    T_surface: np.ndarray = field(default_factory=lambda: np.array([]))

    # Aerodynamic heating (극초음속)
    q_dot_stagnation: float = 0.0   # W/m² (체류점 열유속)

    def __str__(self):
        return (
            f"[{self.phase.value}] M={self.mach:.3f}, h={self.altitude/1000:.1f} km\n"
            f"  Cd: 파동={self.Cd_wave:.4f}, 마찰={self.Cd_friction:.4f}, "
            f"기저={self.Cd_base:.4f}, 합계={self.Cd_total:.4f}\n"
            f"  동압 q={self.dynamic_pressure/1000:.2f} kPa, 항력={self.drag_force/1000:.1f} kN"
        )


class AeroAnalyzer:
    """
    발사체 공기역학 해석기 (Launch Vehicle Aerodynamic Analyzer)
    
    뉴턴-Newtonian 충격파 이론 + DATCOM 경험식 기반 공기역학 특성 계산
    """

    GAMMA = 1.4

    def __init__(self, geometry: VehicleGeometry, isa: Optional[ISAModel] = None):
        self.geo = geometry
        self.isa = isa or ISAModel()

    def analyze(self, mach: float, altitude: float) -> AeroAnalysisResult:
        """
        주어진 마하수와 고도에서의 공기역학 특성 계산
        Compute aerodynamic characteristics at given Mach and altitude.
        """
        atm = self.isa.get_state(altitude)
        velocity = mach * atm.speed_of_sound
        q = 0.5 * atm.density * velocity ** 2

        phase = self._classify_phase(mach, q, altitude)

        # 항력계수 성분 계산
        Cd_wave, Cd_base, shock_features = self._wave_drag(mach, altitude)
        Cd_fric = self._friction_drag(mach, atm, velocity)

        Cd_total = Cd_wave + Cd_fric + Cd_base
        drag = Cd_total * q * self.geo.frontal_area

        # 표면 압력 분포
        x_surf, Cp_surf, T_surf = self._surface_distributions(mach, atm)

        # 극초음속 공력 가열 (Lees 1956)
        q_dot = self._aero_heating(mach, atm) if mach > 5.0 else 0.0

        return AeroAnalysisResult(
            mach=mach, altitude=altitude, velocity=velocity, phase=phase,
            atm=atm,
            Cd_wave=Cd_wave, Cd_friction=Cd_fric, Cd_base=Cd_base,
            Cd_total=Cd_total,
            dynamic_pressure=q, drag_force=drag,
            shock_features=shock_features,
            x_surface=x_surf, Cp_surface=Cp_surf, T_surface=T_surf,
            q_dot_stagnation=q_dot,
        )

    def _classify_phase(self, mach: float, q: float,
                         altitude: float) -> FlightPhase:
        """비행 단계 분류 (Classify flight phase)"""
        if mach < 0.01:
            return FlightPhase.PRE_LAUNCH
        elif mach < 0.8:
            return FlightPhase.SUBSONIC
        elif mach < 1.2:
            # MaxQ는 동압 기준: Falcon 9 기준 ~85 kPa 근방
            if q > 50000:
                return FlightPhase.MAX_Q
            return FlightPhase.TRANSONIC
        elif mach < 5.0:
            return FlightPhase.SUPERSONIC
        else:
            return FlightPhase.HYPERSONIC

    def _wave_drag(self, mach: float, altitude: float):
        """
        파동 항력 계산 (Wave drag estimation)
        
        참고: Hoerner (1965) Ch. 16; Anderson (2003) Ch. 14
        """
        features = []
        g = self.GAMMA
        geo = self.geo

        nose_angle = geo.nose_half_angle
        nose_type  = "ogive"

        # 두부 충격파 특성
        if mach >= 1.0:
            try:
                shock = oblique_shock(mach, nose_angle, weak=True)
                features.append(ShockFeature(
                    feature_type="oblique_shock" if not shock.is_detached else "normal_shock",
                    x_location=0.0,
                    angle=shock.beta,
                    M_upstream=mach,
                    M_downstream=shock.M2,
                    pressure_ratio=shock.p2_p1,
                    temperature_ratio=shock.T2_T1,
                    label=f"두부 충격파 β={shock.beta:.1f}°"
                ))

                # Hammerhead 어깨 부위 특성
                if "hammerhead" in geo.name.lower() or "falcon" in geo.name.lower():
                    # 어깨 팽창파 (Shoulder expansion)
                    shoulder_x = geo.x_profile[
                        np.argmax(geo.r_profile)] if len(geo.r_profile) else 0
                    if mach > 1.2:
                        expansion_angle = 15.0  # Hammerhead shoulder expansion ~15°
                        try:
                            exp = prandtl_meyer_expansion(mach, expansion_angle)
                            features.append(ShockFeature(
                                feature_type="expansion_fan",
                                x_location=shoulder_x,
                                angle=expansion_angle,
                                M_upstream=mach,
                                M_downstream=exp.M2,
                                pressure_ratio=exp.p2_p1,
                                temperature_ratio=exp.T2_T1,
                                label=f"Hammerhead 팽창파 Δθ={expansion_angle}°"
                            ))
                            # 재압축 충격파
                            if exp.M2 > 1.0 and nose_angle > 0:
                                rshock = oblique_shock(exp.M2, nose_angle * 0.5, weak=True)
                                features.append(ShockFeature(
                                    feature_type="oblique_shock",
                                    x_location=shoulder_x + 2.0,
                                    angle=rshock.beta,
                                    M_upstream=exp.M2,
                                    M_downstream=rshock.M2,
                                    pressure_ratio=rshock.p2_p1,
                                    temperature_ratio=rshock.T2_T1,
                                    label=f"재압축 충격파"
                                ))
                        except Exception:
                            pass

            except Exception:
                pass

        # 기저 팽창파
        if mach > 0.5:
            base_x = geo.total_length
            Cd_base = self._base_drag_coefficient(mach)
            features.append(ShockFeature(
                feature_type="expansion_fan",
                x_location=base_x,
                angle=20.0,
                M_upstream=mach,
                M_downstream=mach * 1.1,
                pressure_ratio=0.5 + 0.5 / mach,
                temperature_ratio=0.9,
                label=f"기저 팽창파 (기저항력 기여)"
            ))

        # 파동항력계수 계산
        Cd_wave = wave_drag_coefficient(
            mach, nose_angle,
            base_area=geo.base_area,
            nose_area=np.pi * (geo.r_profile[10] ** 2) if len(geo.r_profile) > 10 else geo.frontal_area,
            nose_type=nose_type
        )
        Cd_base = self._base_drag_coefficient(mach)

        return Cd_wave, Cd_base, features

    def _base_drag_coefficient(self, mach: float) -> float:
        """
        기저 항력계수 (Base drag coefficient)
        
        Reference: Hoerner (1965), Ch. 16, Fig. 24:
          Cd_base = f(M) × (A_base / A_ref)
        """
        A_ratio = self.geo.base_area / self.geo.frontal_area
        if mach < 0.6:
            Cd_b = 0.12
        elif mach < 1.0:
            # 천음속 증가 (Transonic rise)
            Cd_b = 0.12 + 0.3 * (mach - 0.6) ** 2
        elif mach < 2.0:
            Cd_b = 0.28 / mach ** 1.3
        else:
            Cd_b = 0.15 / mach ** 1.0
        return Cd_b * A_ratio

    def _friction_drag(self, mach: float, atm: AtmosphericState,
                        velocity: float) -> float:
        """
        마찰 항력계수 (Skin friction drag)
        
        Reference: White (2006); Van Driest (1956); Eckert (1956)
        """
        if velocity < 0.1:
            return 0.0
        Re = atm.density * velocity * self.geo.total_length / atm.dynamic_viscosity
        return friction_drag_coefficient(
            mach, Re,
            wetted_area=self.geo.wetted_area,
            reference_area=self.geo.frontal_area
        )

    def _surface_distributions(self, mach: float,
                                 atm: AtmosphericState
                                 ) -> tuple:
        """
        표면 압력/온도 분포 계산
        Surface pressure & temperature coefficient distributions.
        
        Method:
          - 아음속: 선형 Cp 분포 근사
          - 초음속: Newton 충격이론 (Modified Newtonian, Lees 1956)
          
        Reference:
          - Lees, L. (1956). "Laminar heat transfer over blunt-nosed bodies."
          - Bertin, J.J. (1994). Hypersonic Aerothermodynamics. AIAA Education.
        """
        x = self.geo.x_profile
        r = self.geo.r_profile
        g = self.GAMMA

        # 국소 경사각
        dr = np.gradient(r, x)
        theta = np.arctan(np.abs(dr))   # 국소 표면 기울기 [rad]

        if mach < 0.8:
            # 아음속: 간단한 선형 근사 Cp
            Cp = -2 * theta / np.pi
            T_ratio = np.ones_like(Cp)

        elif mach < 1.2:
            # 천음속: 임계 Cp 혼합
            Cp_critical = -2 / (g * mach ** 2) * (
                (2 / (g + 1) + (g - 1) / (g + 1) * mach ** 2) ** (g / (g - 1)) - 1)
            Cp = Cp_critical * (1 - np.cos(2 * theta))
            T_ratio = 1 + (g - 1) / 2 * mach ** 2 * np.sin(theta) ** 2

        elif mach >= 5.0:
            # 극초음속 수정 뉴턴 이론
            # Modified Newtonian: Cp = Cp_max * sin²(θ)
            # Reference: Lees (1956), Bertin (1994)
            ns = normal_shock(mach)
            Cp_max = 2 / (g * mach ** 2) * (ns.p2_p1 - 1)
            Cp = Cp_max * np.sin(theta) ** 2
            T_ratio = 1 + (g - 1) / 2 * mach ** 2 * np.sin(theta) ** 2

        else:
            # 초음속: Newtonian 이론
            Cp = 2 * np.sin(theta) ** 2  # 고전 뉴턴 이론 Cp
            T_ratio = 1 + (g - 1) / 2 * mach ** 2 * np.sin(theta) ** 2

        T_surface = atm.temperature * T_ratio
        return x, Cp, T_surface

    def _aero_heating(self, mach: float, atm: AtmosphericState) -> float:
        """
        극초음속 공력 가열 - 체류점 열유속 (Stagnation point heat flux)
        Fay-Riddell correlation (simplified).
        
        Reference:
          - Fay, J.A. & Riddell, F.R. (1958). "Theory of stagnation point heat
            transfer in dissociated air." J. Aero. Sci. 25(2), 73–85.
          - Re-entry vehicle heating estimation: 간략식 사용
        
        Returns: q_dot [W/m²]
        """
        rho = atm.density
        V   = mach * atm.speed_of_sound
        R_nose = self.geo.r_profile[5] if len(self.geo.r_profile) > 5 else 0.5   # 코끝 반경 [m]
        R_nose = max(R_nose, 0.01)  # 최솟값 보정

        # Sutton-Graves correlation (Sutton, K. & Graves, R.A., 1971)
        # q_dot = k * sqrt(rho/R_nose) * V^3
        # k = 1.741e-4 (earth atmosphere, SI units)
        k = 1.741e-4
        q_dot = k * np.sqrt(rho / R_nose) * V ** 3
        return q_dot


class TrajectoryAnalyzer:
    """
    발사 궤도 전체 분석기 (Full trajectory analysis)
    다양한 마하수 구간에서 공기역학 특성 스윕
    """

    def __init__(self, geometry: VehicleGeometry, isa: Optional[ISAModel] = None):
        self.analyzer = AeroAnalyzer(geometry, isa)
        self.isa = isa or ISAModel()

    def mach_sweep(self, mach_range: np.ndarray,
                   altitude_profile: Optional[np.ndarray] = None
                   ) -> List[AeroAnalysisResult]:
        """
        마하수 스윕 (Mach sweep for drag polar)
        
        Args:
            mach_range       : 마하수 배열
            altitude_profile : 고도 배열 (None이면 전형적 궤도 사용)
        """
        if altitude_profile is None:
            # 전형적인 LEO 발사 고도 프로파일
            altitude_profile = self._typical_altitude_profile(mach_range)

        results = []
        for M, h in zip(mach_range, altitude_profile):
            try:
                res = self.analyzer.analyze(M, h)
                results.append(res)
            except Exception as e:
                print(f"  [경고] M={M:.2f}, h={h:.0f}m 계산 실패: {e}")
        return results

    def _typical_altitude_profile(self, machs: np.ndarray) -> np.ndarray:
        """
        전형적인 LEO 발사 고도-마하수 관계
        Typical altitude-Mach relationship for LEO launch.
        
        Reference: NASA SP-2010-3403, "NASA Systems Engineering Handbook"
        Approximate profile based on Falcon 9 flight data.
        """
        altitudes = np.zeros_like(machs, dtype=float)
        for i, M in enumerate(machs):
            if M < 1.0:
                # 0~10 km (aero dynamic pressure phase)
                altitudes[i] = 15000 * M
            elif M < 3.0:
                # 10~50 km
                altitudes[i] = 15000 + 35000 * (M - 1.0) / 2.0
            elif M < 10.0:
                # 50~120 km
                altitudes[i] = 50000 + 70000 * (M - 3.0) / 7.0
            else:
                altitudes[i] = 120000
        return altitudes


if __name__ == "__main__":
    from geometry.shapes import make_falcon9, make_generic_rocket, make_optimized_rocket

    print("=== 공기역학 해석 검증 ===\n")

    for make_fn in [make_generic_rocket, make_falcon9, make_optimized_rocket]:
        geo = make_fn()
        analyzer = AeroAnalyzer(geo)
        print(f"\n--- {geo.name} ---")
        for M in [0.5, 0.9, 1.2, 2.0, 5.0, 8.0]:
            h = 20000 if M < 2 else 50000
            res = analyzer.analyze(M, h)
            print(res)
