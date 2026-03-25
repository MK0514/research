"""
발사체 형상 정의 모듈
Launch Vehicle Geometry Definition

References:
  - SpaceX Falcon 9 User's Guide, Rev 2.0 (2021)
    https://www.spacex.com/media/falcon-9-users-guide-2021-09.pdf
  - Hoerner, S.F. (1965). Fluid-Dynamic Drag. Published by the author.
  - Barrowman, J.S. (1966). "The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles." M.S. Thesis, Catholic
    University of America. (OpenRocket 기반 안정성 계산)
  - NASA SP-8012 (1971). "Natural Vibration Modal Analysis." (형상 기준)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class NoseType(Enum):
    """두부 형상 타입 (Nose cone shape)"""
    CONICAL   = "conical"        # 원추형
    OGIVE     = "ogive"          # 오지브형 (가장 일반적)
    VON_KARMAN = "von_karman"    # Von Kármán (최소 파동항력)
    POWER     = "power"          # Power series
    BLUNT     = "blunt"          # 뭉뚝형 (유인 캡슐 등)
    HAMMERHEAD = "hammerhead"    # Falcon 9 형상 (Hammerhead fairing)
    PARABOLIC = "parabolic"      # 포물선형


@dataclass
class GeometrySection:
    """발사체 단면 (Vehicle cross-section)"""
    name: str
    length: float        # m
    diameter_base: float # m (하단 직경)
    diameter_top: float  # m (상단 직경)
    nose_type: Optional[NoseType] = None
    n_panels: int = 100  # 형상 이산화 포인트 수


@dataclass
class VehicleGeometry:
    """
    발사체 전체 형상 (Complete vehicle geometry)
    
    좌표계: 두부(Nose) = x=0, 후미(Base) = x=L
    Coordinate: Nose at x=0, Base at x=L
    """
    name: str
    sections: List[GeometrySection]
    fins: Optional[dict] = None  # 핀/그리드핀 정보

    def __post_init__(self):
        self._build_profile()

    def _build_profile(self):
        """형상 프로파일 계산 (Compute geometry profile arrays)"""
        x_all, r_all = [], []
        x_cursor = 0.0

        for sec in self.sections:
            xi = np.linspace(0, sec.length, sec.n_panels)
            r_base = sec.diameter_base / 2
            r_top  = sec.diameter_top / 2

            if sec.nose_type is None or sec.nose_type == NoseType.CONICAL:
                # Linear taper
                ri = r_top + (r_base - r_top) * xi / sec.length

            elif sec.nose_type == NoseType.OGIVE:
                # Secant ogive (Barrowman 1966)
                R = r_base
                L = sec.length
                if R < 1e-9:
                    # 두부 끝점이 R=0이면 선형으로 처리
                    ri = r_top + (r_base - r_top) * xi / L
                else:
                    rho_r = (R ** 2 + L ** 2) / (2 * R)
                    ri = np.sqrt(np.maximum(rho_r ** 2 - (L - xi) ** 2, 0)) + R - rho_r
                    ri = np.clip(ri, 0, R)

            elif sec.nose_type == NoseType.VON_KARMAN:
                # Haack series, C=0 (Von Kármán ogive)
                # Minimizes wave drag for given length and base radius
                # Reference: Sears, W.R. (1947). "On projectiles of minimum
                # wave drag." Quarterly of Applied Mathematics, 4(4), 361–366.
                L = sec.length
                R = r_base
                theta = np.arccos(1 - 2 * xi / L)
                ri = R / np.sqrt(np.pi) * np.sqrt(
                    theta - np.sin(2 * theta) / 2)

            elif sec.nose_type == NoseType.PARABOLIC:
                # Parabolic nose: r = R*(2*x/L - (x/L)²) for K=1 (tangent ogive)
                L = sec.length
                R = r_base
                t = xi / L
                ri = R * (2 * t - t ** 2)

            elif sec.nose_type == NoseType.HAMMERHEAD:
                # Falcon 9 Hammerhead 형상
                # 페어링(fairing)이 동체보다 큰 직경을 가짐
                # Reference: SpaceX Falcon 9 User's Guide (2021) §3.1
                L = sec.length
                R_fairing = r_top       # 페어링 반경 (큰 쪽)
                R_body    = r_base      # 동체 반경 (작은 쪽)
                # Smooth conical transition with slight bulge at base
                t  = xi / L
                ri = R_fairing * (1 - t) + R_body * t + \
                     0.05 * R_fairing * np.sin(np.pi * t)

            elif sec.nose_type == NoseType.BLUNT:
                # Spherical cap + cone
                R = r_base
                L = sec.length
                R_sphere = R * 0.5  # 구형 캡 반경 = 0.5 * base radius
                x_sphere = np.sqrt(R_sphere ** 2 - (R_sphere - R) ** 2) \
                           if R_sphere > R else 0.0
                ri = np.where(
                    xi < x_sphere,
                    np.sqrt(np.maximum(R_sphere ** 2 - (xi - 0) ** 2, 0)),
                    R * (xi - x_sphere) / (L - x_sphere)
                )
            else:
                # Fallback: linear
                ri = r_top + (r_base - r_top) * xi / sec.length

            x_all.append(x_cursor + xi)
            r_all.append(ri)
            x_cursor += sec.length

        self.x_profile = np.concatenate(x_all)
        self.r_profile = np.concatenate(r_all)
        self.total_length = x_cursor

    @property
    def max_diameter(self) -> float:
        return 2 * np.max(self.r_profile)

    @property
    def base_diameter(self) -> float:
        return 2 * self.r_profile[-1]

    @property
    def nose_half_angle(self) -> float:
        """유효 두부 반각 (Effective nose half-angle) [deg]"""
        dx = self.x_profile[10] - self.x_profile[0]
        dr = self.r_profile[10] - self.r_profile[0]
        if dx <= 0:
            return 0.0
        return np.degrees(np.arctan(abs(dr / dx)))

    @property
    def fineness_ratio(self) -> float:
        """세장비 L/D (Fineness ratio = length / max diameter)"""
        return self.total_length / self.max_diameter

    @property
    def frontal_area(self) -> float:
        """전면 면적 [m²] (Frontal/reference area)"""
        return np.pi * (self.max_diameter / 2) ** 2

    @property
    def base_area(self) -> float:
        """기저 면적 [m²]"""
        return np.pi * (self.base_diameter / 2) ** 2

    @property
    def wetted_area(self) -> float:
        """
        습윤 면적 [m²] (Wetted area, surface of revolution)
        Numerical integration of 2π·r·ds along profile
        """
        dx = np.diff(self.x_profile)
        dr = np.diff(self.r_profile)
        ds = np.sqrt(dx ** 2 + dr ** 2)
        r_mid = (self.r_profile[:-1] + self.r_profile[1:]) / 2
        return float(np.sum(2 * np.pi * r_mid * ds))

    @property
    def volume(self) -> float:
        """체적 [m³] (Volume by disk integration)"""
        dx = np.diff(self.x_profile)
        r_mid = (self.r_profile[:-1] + self.r_profile[1:]) / 2
        return float(np.sum(np.pi * r_mid ** 2 * dx))

    def local_slope(self) -> np.ndarray:
        """국소 표면 기울기 [deg] (Local surface slope dθ/dx)"""
        dr = np.gradient(self.r_profile, self.x_profile)
        return np.degrees(np.arctan(np.abs(dr)))

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  전체 길이    : {self.total_length:.2f} m",
            f"  최대 직경    : {self.max_diameter:.3f} m",
            f"  세장비 (L/D) : {self.fineness_ratio:.2f}",
            f"  두부 반각    : {self.nose_half_angle:.2f}°",
            f"  전면 면적    : {self.frontal_area:.4f} m²",
            f"  습윤 면적    : {self.wetted_area:.2f} m²",
            f"  체적         : {self.volume:.2f} m³",
        ]
        return "\n".join(lines)


# ─── 사전 정의 발사체 형상 ────────────────────────────────────────────────────

def make_generic_rocket(
    total_length: float = 60.0,
    max_diameter: float = 3.7,
    nose_length_ratio: float = 0.12,
    nose_type: NoseType = NoseType.OGIVE,
) -> VehicleGeometry:
    """
    일반적인 우주발사체 형상 (Generic launch vehicle)
    
    비율 기반으로 스케일 가능.
    """
    L = total_length
    D = max_diameter
    R = D / 2

    nose_L   = L * nose_length_ratio
    body1_L  = L * 0.50   # 1단
    interstage_L = L * 0.05
    body2_L  = L * 0.28   # 2단
    fairing_L = L * 0.05  # 페어링

    return VehicleGeometry(
        name="Generic Launch Vehicle",
        sections=[
            GeometrySection("Nose Cone",  nose_L,       R * 0.0, R,   NoseType.OGIVE),
            GeometrySection("Stage 2",    body2_L,      R,       R,   None),
            GeometrySection("Interstage", interstage_L, R,       R * 0.9, None),
            GeometrySection("Stage 1",    body1_L,      R * 0.9, R * 0.9, None),
        ]
    )


def make_falcon9() -> VehicleGeometry:
    """
    SpaceX Falcon 9 Block 5 (Hammerhead) 형상
    
    실측 치수:
    - 전체 길이: 70.0 m (페어링 포함)
    - 동체 직경: 3.66 m (12 ft)
    - 페어링 직경: 5.2 m
    - 페어링 길이: 13.1 m
    - 1단 길이: ~42.6 m
    - 2단 길이: ~12.6 m
    
    Reference:
      - SpaceX Falcon 9 User's Guide (2021), Table 2.1, §3.1
      - McDowell, J. (2020). "The Falcon 9 v1.1 and v1.2 (Full Thrust) launch
        vehicle." Spaceflight, 62(10). [공개 치수 정보]
      - Wade, M. (ed.). Encyclopedia Astronautica: Falcon 9.
    
    Hammerhead 특징:
      - 페어링이 동체보다 큰 직경 → 천음속 영역에서 충격파 구조 복잡
      - MaxQ에서 인터스테이지 부근 불안정 충격파-경계층 상호작용 발생
      - 페어링 어깨(shoulder) 부위: Prandtl-Meyer 팽창 후 재압축 충격파
    """
    R_body    = 3.66 / 2   # 동체 반경 [m]
    R_fairing = 5.20 / 2   # 페어링 반경 [m]

    return VehicleGeometry(
        name="SpaceX Falcon 9 Block 5 (Hammerhead)",
        sections=[
            # 페어링 두부 (Payload fairing nose)
            GeometrySection("Fairing Nose",      4.0,  R_fairing * 0.0, R_fairing, NoseType.OGIVE),
            # 페어링 실린더 (Fairing cylinder)
            GeometrySection("Fairing Cylinder",  9.1,  R_fairing, R_fairing, None),
            # Hammerhead 어깨 전이부 (Shoulder transition - key aerodynamic feature)
            GeometrySection("Hammerhead Shoulder", 2.0, R_fairing, R_body, NoseType.HAMMERHEAD),
            # 2단 동체 (2nd stage)
            GeometrySection("Stage 2",          12.6, R_body, R_body, None),
            # 인터스테이지 (Interstage)
            GeometrySection("Interstage",        1.3,  R_body, R_body * 0.95, None),
            # 1단 동체 (1st stage)
            GeometrySection("Stage 1",          41.2, R_body * 0.95, R_body * 0.95, None),
        ],
        fins={
            "type": "grid_fin",     # 그리드 핀 (Falcon 9 grid fins)
            "count": 4,
            "span": 1.5,            # m (deployed span)
            "chord": 0.5,           # m
            "position_from_base": 0.5,
        }
    )


def make_optimized_rocket(
    fineness_ratio: float = 15.0,
    nose_type: NoseType = NoseType.VON_KARMAN,
    base_diameter: float = 3.66,
) -> VehicleGeometry:
    """
    최적화 발사체 형상 (Aerodynamically optimized vehicle)
    
    최적화 기준:
    1. Von Kármán 두부 → 초음속 파동항력 최소
    2. 높은 세장비 (L/D=15) → 조파항력 감소
    3. 균일 직경 동체 → 충격파-경계층 상호작용 최소
    4. 후방 보트테일 → 기저항력 감소
    
    References:
      - Sears, W.R. (1947). "On projectiles of minimum wave drag."
        Quarterly of Applied Mathematics, 4(4), 361–366.
      - Adams, M.C. (1953). "Determination of shapes of boattail bodies of
        revolution for minimum wave drag." NACA TN 2550.
      - Haack, W. (1941). Geschossformen kleinsten Wellenwiderstandes.
        Lilienthal-Gesellschaft für Luftfahrtforschung, Report 139.
    """
    D = base_diameter
    R = D / 2
    L = fineness_ratio * D

    nose_L   = L * 0.15
    body_L   = L * 0.70
    boattail_L = L * 0.10
    stage2_L = L * 0.05

    R_boattail = R * 0.75  # 후방 보트테일 축소 직경

    return VehicleGeometry(
        name=f"Optimized LV (Von Kármán Nose, L/D={fineness_ratio:.0f})",
        sections=[
            GeometrySection("Von Kármán Nose",   nose_L,    0.0, R,          NoseType.VON_KARMAN),
            GeometrySection("Stage 2 Body",       stage2_L,  R,   R,          None),
            GeometrySection("Stage 1 Body",       body_L,    R,   R,          None),
            GeometrySection("Boattail",           boattail_L, R, R_boattail, None),
        ]
    )


# ─── STL 파일 로더는 별도 모듈 ────────────────────────────────────────────────

if __name__ == "__main__":
    for make_fn in [make_generic_rocket, make_falcon9, make_optimized_rocket]:
        geo = make_fn()
        print(geo.summary())
        print()
