"""
충격파 및 팽창파 해석 모듈
Shock Wave & Prandtl-Meyer Expansion Fan Analysis

References:
  - Anderson, J.D. (2003). Modern Compressible Flow, 3rd ed. McGraw-Hill.
    Ch.4 (Normal Shocks), Ch.9 (Oblique Shocks), Ch.4.6 (Prandtl-Meyer)
  - Shapiro, A.H. (1953). The Dynamics and Thermodynamics of Compressible Fluid Flow.
    Ronald Press Company, Vol.1.
  - NACA TN 1135: Equations, Tables, and Charts for Compressible Flow (1953)
"""

import numpy as np
from scipy.optimize import brentq, fsolve
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShockResult:
    """경사충격파 결과 (Oblique Shock Result)"""
    M1: float        # 업스트림 마하수
    theta: float     # 웨지 반각 [deg] (deflection angle)
    beta: float      # 충격각 [deg] (shock angle)
    M2: float        # 다운스트림 마하수
    p2_p1: float     # 압력비 (pressure ratio)
    rho2_rho1: float # 밀도비 (density ratio)
    T2_T1: float     # 온도비 (temperature ratio)
    p0_ratio: float  # 전압 비율 (stagnation pressure ratio)
    is_detached: bool  # 박리 충격파 여부 (detached shock)


@dataclass
class NormalShockResult:
    """수직충격파 결과 (Normal Shock Result)"""
    M1: float
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p0_ratio: float


@dataclass
class ExpansionResult:
    """팽창파 결과 (Prandtl-Meyer Expansion Result)"""
    M1: float
    M2: float
    theta: float     # 팽창각 [deg]
    p2_p1: float
    T2_T1: float
    rho2_rho1: float
    nu1: float       # Prandtl-Meyer 함수 값 (upstream)
    nu2: float       # Prandtl-Meyer 함수 값 (downstream)


GAMMA = 1.4   # 공기 비열비 (ratio of specific heats for air)


# ─── Normal Shock Relations ────────────────────────────────────────────────

def normal_shock(M1: float) -> NormalShockResult:
    """
    수직 충격파 관계식
    Rankine-Hugoniot relations for normal shock.
    
    Reference: Anderson (2003), Eq. 3.57–3.65
    
    M1: upstream Mach number (must be ≥ 1)
    """
    if M1 < 1.0:
        raise ValueError(f"Normal shock requires M1 ≥ 1 (got M1={M1:.3f})")

    g = GAMMA
    M1sq = M1 ** 2

    # Downstream Mach (Anderson Eq. 3.57)
    M2sq = (1 + (g - 1) / 2 * M1sq) / (g * M1sq - (g - 1) / 2)
    M2   = np.sqrt(np.maximum(M2sq, 0.0))

    # Pressure ratio p2/p1 (Rankine-Hugoniot, Anderson Eq. 3.57)
    p2_p1 = 1 + 2 * g / (g + 1) * (M1sq - 1)

    # Density ratio rho2/rho1 (Anderson Eq. 3.57)
    rho2_rho1 = (g + 1) * M1sq / (2 + (g - 1) * M1sq)

    # Temperature ratio T2/T1 via ideal gas: T2/T1 = (p2/p1)*(rho1/rho2)
    T2_T1 = p2_p1 / rho2_rho1

    # Stagnation pressure ratio (Anderson Eq. 3.63)
    p0_ratio = ((( (g + 1) * M1sq) / (2 + (g - 1) * M1sq)) ** (g / (g - 1)) *
                (1 / p2_p1) ** (1 / (g - 1)) *
                ((2 * g * M1sq / (g + 1) - (g - 1) / (g + 1))) ** (-1 / (g - 1)))

    return NormalShockResult(M1, M2, p2_p1, rho2_rho1, T2_T1, p0_ratio)


# ─── Oblique Shock Relations ───────────────────────────────────────────────

def _beta_from_theta_M(theta_deg: float, M1: float,
                       weak: bool = True) -> Optional[float]:
    """
    theta-beta-M 관계식으로 충격각 beta 계산
    Solve for shock angle beta given deflection angle theta and M1.
    
    Reference: Anderson (2003), Eq. 9.13 (theta-beta-M relation):
      tan(theta) = 2*cot(beta) * (M1²·sin²(beta) - 1) /
                   (M1²*(γ + cos(2β)) + 2)
    """
    theta = np.radians(theta_deg)
    g = GAMMA

    def tbm(beta_rad):
        b = beta_rad
        sb2 = np.sin(b) ** 2
        num = 2 / np.tan(b) * (M1 ** 2 * sb2 - 1)
        den = M1 ** 2 * (g + np.cos(2 * b)) + 2
        return np.tan(theta) - num / den

    # Minimum shock angle = Mach angle mu = arcsin(1/M1)
    mu = np.degrees(np.arcsin(1.0 / M1)) if M1 >= 1 else 0.0
    beta_min = np.radians(mu + 0.01)
    beta_max = np.radians(89.9)

    # Find theta_max (detachment angle) to check feasibility
    try:
        theta_max = _theta_max(M1)
        if theta_deg > theta_max:
            return None   # Detached shock (박리)
    except Exception:
        pass

    try:
        if weak:
            # Weak solution: search [mu, beta_sonic]
            beta_sonic = _beta_at_sonic(M1)
            root = brentq(tbm, beta_min, np.radians(beta_sonic), xtol=1e-8)
        else:
            # Strong solution: search [beta_sonic, 90°]
            beta_sonic = _beta_at_sonic(M1)
            root = brentq(tbm, np.radians(beta_sonic), beta_max, xtol=1e-8)
        return np.degrees(root)
    except Exception:
        return None


def _theta_max(M1: float) -> float:
    """최대 편향각(박리 한계) 계산 (Max deflection angle / detachment)"""
    g = GAMMA

    def neg_theta(beta_rad):
        b = beta_rad
        sb2 = np.sin(b) ** 2
        num = 2 / np.tan(b) * (M1 ** 2 * sb2 - 1)
        den = M1 ** 2 * (g + np.cos(2 * b)) + 2
        return -np.arctan(num / den)

    mu = np.arcsin(1.0 / M1)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(neg_theta, bounds=(mu, np.pi / 2 - 0.01),
                          method='bounded')
    return np.degrees(-res.fun)


def _beta_at_sonic(M1: float) -> float:
    """
    다운스트림 M2=1이 되는 충격각 (Sonic point = boundary weak/strong)
    """
    g = GAMMA

    def sonic_condition(beta_rad):
        Mn1 = M1 * np.sin(beta_rad)
        if Mn1 < 1:
            return 1.0
        M2sq = (1 + (g - 1) / 2 * Mn1 ** 2) / (g * Mn1 ** 2 - (g - 1) / 2)
        beta = beta_rad
        theta = np.arctan(2 / np.tan(beta) * (M1 ** 2 * np.sin(beta) ** 2 - 1) /
                          (M1 ** 2 * (g + np.cos(2 * beta)) + 2))
        M2 = np.sqrt(M2sq) / np.sin(beta - theta)
        return M2 - 1.0

    mu = np.arcsin(1.0 / M1) + 1e-4
    try:
        return np.degrees(brentq(sonic_condition, mu, np.pi / 2 - 0.01, xtol=1e-8))
    except Exception:
        return 90.0


def oblique_shock(M1: float, theta_deg: float,
                  weak: bool = True) -> ShockResult:
    """
    경사 충격파 계산
    Oblique shock relations.
    
    Reference: Anderson (2003), Ch. 9
      - theta-beta-M: Eq. 9.13
      - Normal component Mach: M_n1 = M1*sin(beta)
      - Downstream Mach: Eq. 9.12
    
    Args:
        M1        : upstream Mach number
        theta_deg : wedge half-angle (deflection) [deg]
        weak      : True = weak shock solution
    """
    if M1 < 1.0:
        raise ValueError("Oblique shock requires supersonic flow (M1 ≥ 1)")

    beta = _beta_from_theta_M(theta_deg, M1, weak)
    is_detached = beta is None

    if is_detached:
        # 박리 충격파: 수직충격파로 근사 (Detached → use normal shock as approximation)
        ns = normal_shock(M1)
        return ShockResult(M1, theta_deg, 90.0, ns.M2, ns.p2_p1,
                           ns.rho2_rho1, ns.T2_T1, ns.p0_ratio, True)

    g = GAMMA
    beta_rad  = np.radians(beta)
    theta_rad = np.radians(theta_deg)

    Mn1 = M1 * np.sin(beta_rad)
    ns  = normal_shock(Mn1)

    M2 = np.sqrt(ns.M2 ** 2 + (M1 * np.cos(beta_rad) /
                               (1 + (g - 1) / 2 * ns.M2 ** 2 * np.sin(beta_rad - theta_rad) ** 2)) ** 2) \
         if False else ns.M2 / np.sin(beta_rad - theta_rad)

    return ShockResult(
        M1=M1, theta=theta_deg, beta=beta,
        M2=M2, p2_p1=ns.p2_p1, rho2_rho1=ns.rho2_rho1,
        T2_T1=ns.T2_T1, p0_ratio=ns.p0_ratio,
        is_detached=False
    )


# ─── Prandtl-Meyer Expansion ───────────────────────────────────────────────

def prandtl_meyer_function(M: float) -> float:
    """
    Prandtl-Meyer 함수 ν(M) 계산 [radians]
    
    Reference: Anderson (2003), Eq. 4.44:
      ν(M) = √((γ+1)/(γ-1)) · arctan(√((γ-1)/(γ+1)·(M²-1)))
             - arctan(√(M²-1))
    """
    if M < 1.0:
        return 0.0
    g = GAMMA
    term1 = np.sqrt((g + 1) / (g - 1)) * np.arctan(
        np.sqrt((g - 1) / (g + 1) * (M ** 2 - 1)))
    term2 = np.arctan(np.sqrt(M ** 2 - 1))
    return term1 - term2


def _M_from_prandtl_meyer(nu_rad: float) -> float:
    """ν 값으로부터 마하수 역산 (Inverse Prandtl-Meyer)"""
    nu_max = prandtl_meyer_function(100.0)
    nu_rad = np.clip(nu_rad, 0, nu_max)

    def eq(M):
        return prandtl_meyer_function(M[0]) - nu_rad

    M_init = max(1.01, 1 + nu_rad)
    sol = fsolve(eq, [M_init], full_output=False)
    return float(sol[0])


def prandtl_meyer_expansion(M1: float, theta_deg: float) -> ExpansionResult:
    """
    Prandtl-Meyer 팽창파 계산
    Isentropic expansion fan relations.
    
    Reference: Anderson (2003), Sec. 4.5–4.6; NACA TN 1135
    
    Args:
        M1        : upstream Mach number (≥ 1)
        theta_deg : expansion angle [deg] (positive = expansion)
    """
    if M1 < 1.0:
        raise ValueError("Expansion fan requires supersonic flow (M1 ≥ 1)")

    g = GAMMA
    nu1 = prandtl_meyer_function(M1)
    nu2 = nu1 + np.radians(theta_deg)
    M2  = _M_from_prandtl_meyer(nu2)

    # Isentropic relations (Anderson Eq. 4.74)
    def isentropic_T_ratio(M):
        return 1 + (g - 1) / 2 * M ** 2

    T_ratio = isentropic_T_ratio(M1) / isentropic_T_ratio(M2)
    p_ratio = T_ratio ** (g / (g - 1))
    rho_ratio = T_ratio ** (1 / (g - 1))

    return ExpansionResult(
        M1=M1, M2=M2, theta=theta_deg,
        p2_p1=p_ratio, T2_T1=T_ratio, rho2_rho1=rho_ratio,
        nu1=np.degrees(nu1), nu2=np.degrees(nu2)
    )


# ─── Conical Shock (Taylor-Maccoll) ───────────────────────────────────────

def mach_angle(M: float) -> float:
    """마하각 μ = arcsin(1/M) [degrees]"""
    if M < 1.0:
        return 90.0
    return np.degrees(np.arcsin(1.0 / M))


def bow_shock_standoff(M: float, nose_type: str = "blunt") -> float:
    """
    두부 충격파 이격 거리 추정 (Bow shock standoff distance)
    Normalized by nose radius: Δ/R
    
    Reference:
      - Billig, F.S. (1967). "Shock-wave shapes around spherical and cylindrical-
        nosed bodies." J. Spacecraft and Rockets, 4(6), 822–823.
    """
    if M < 1.0:
        return np.inf
    if nose_type == "blunt":
        # Billig (1967) correlation for blunt nose
        return 0.386 * np.exp(4.67 / M ** 2)
    else:
        return 0.0   # Sharp nose: attached shock


# ─── Drag Coefficient Estimation ──────────────────────────────────────────

def wave_drag_coefficient(M: float, nose_half_angle: float,
                          base_area: float, nose_area: float,
                          nose_type: str = "ogive") -> float:
    """
    파동항력계수 계산 (Wave drag coefficient)
    
    References:
      - Hoerner, S.F. (1965). Fluid-Dynamic Drag. Ch. 16 (Supersonic Drag)
      - NACA RM A53G17: Drag of bodies of revolution in supersonic flow
      - Cronvich, L.L. (1974). "Missile Aerodynamics," AIAA Paper
    
    두 가지 성분:
      1. 선두부 파동항력 (Nose wave drag): oblique shock 기반
      2. 기저 항력 (Base drag): 후류 압력 저하
    """
    if M < 0.8:
        return 0.0   # 아음속 파동항력 무시

    g = GAMMA

    # 1. Nose wave drag
    if M >= 1.0:
        try:
            shock = oblique_shock(M, nose_half_angle, weak=True)
            Cp_nose = 2 / (g * M ** 2) * (shock.p2_p1 - 1)
            if nose_type == "ogive":
                # Ogive nose: lower drag than cone (shape factor ~0.7)
                shape_factor = 0.70
            elif nose_type == "cone":
                shape_factor = 1.00
            elif nose_type == "sphere" or nose_type == "blunt":
                # Blunt body: use normal shock pressure coefficient
                ns = normal_shock(M)
                Cp_nose = 2 / (g * M ** 2) * (ns.p2_p1 - 1)
                shape_factor = 0.85
            else:
                shape_factor = 0.75

            Cd_wave_nose = shape_factor * Cp_nose * (nose_area / base_area) * np.sin(
                np.radians(nose_half_angle))
        except Exception:
            Cd_wave_nose = 0.0
    else:
        Cd_wave_nose = 0.0

    # 2. Base drag (Hoerner 1965, Ch. 16)
    if M < 1.0:
        Cd_base = 0.12
    elif M < 2.0:
        # Supersonic base drag decreases with Mach
        Cd_base = 0.25 / M ** 1.5
    else:
        Cd_base = 0.12 / M ** 1.2

    return Cd_wave_nose + Cd_base


def friction_drag_coefficient(M: float, Re: float,
                               wetted_area: float,
                               reference_area: float) -> float:
    """
    마찰 항력계수 (Skin friction drag coefficient)
    
    References:
      - White, F.M. (2006). Viscous Fluid Flow, 3rd ed. McGraw-Hill.
      - van Driest, E.R. (1956). "The Problem of Aerodynamic Heating."
        Aeronautical Engineering Review, 15(10), 26–41.
      - Eckert, E.R.G. (1956). "Engineering relations for friction and heat
        transfer to surfaces in high velocity flow." J. Aero. Sci. 22, 585–587.
    
    Returns Cf based on reference temperature method (Eckert 1956)
    """
    if Re < 1e4:
        return 0.0

    # 기준 온도법 보정 계수 (Reference temperature correction, Van Driest)
    # T* / T∞ ≈ 0.5 + 0.039*M² (simplified, Van Driest 1956)
    T_ratio = 0.5 + 0.039 * M ** 2 if M > 0 else 1.0
    Re_star = Re / T_ratio  # 보정 레이놀즈수

    # Prandtl 난류 경계층 (Turbulent boundary layer)
    if Re_star > 5e5:
        Cf_local = 0.074 / Re_star ** 0.2   # Schlichting 1/5 power law
    else:
        Cf_local = 1.328 / np.sqrt(Re_star)  # Blasius laminar

    return Cf_local * wetted_area / reference_area


# ─── Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 충격파/팽창파 검증 (Shock/Expansion Verification) ===\n")

    print("[ 수직충격파 (Normal Shock) ]")
    for M in [1.5, 2.0, 3.0, 5.0]:
        r = normal_shock(M)
        print(f"  M1={M}: M2={r.M2:.4f}, p2/p1={r.p2_p1:.4f}, T2/T1={r.T2_T1:.4f}")

    print("\n[ 경사충격파 (Oblique Shock, theta=15°) ]")
    for M in [2.0, 3.0, 5.0]:
        r = oblique_shock(M, 15.0)
        print(f"  M1={M}: beta={r.beta:.2f}°, M2={r.M2:.4f}, p2/p1={r.p2_p1:.4f}")

    print("\n[ Prandtl-Meyer 팽창파 (theta=20°) ]")
    for M in [2.0, 3.0, 5.0]:
        r = prandtl_meyer_expansion(M, 20.0)
        print(f"  M1={M}: M2={r.M2:.4f}, p2/p1={r.p2_p1:.4f}")
