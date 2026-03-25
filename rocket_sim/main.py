"""
우주발사체 공기역학 시뮬레이션 - 메인 실행 파일
Launch Vehicle Aerodynamic Simulation Suite - Main Runner

사용법 (Usage):
  python main.py                    → 전체 분석 실행
  python main.py --vehicle falcon9  → Falcon 9만 분석
  python main.py --stl my_rocket.stl --scale 0.001  → STL 파일 로드 (mm→m)
  python main.py --mach 2.5 --alt 30000  → 특정 마하수/고도 분석

References:
  [1] Anderson, J.D. (2003). Modern Compressible Flow, 3rd ed. McGraw-Hill.
  [2] Hoerner, S.F. (1965). Fluid-Dynamic Drag. Published by the author.
  [3] ISO 2533:1975. Standard Atmosphere.
  [4] SpaceX Falcon 9 User's Guide, Rev 2.0 (2021).
  [5] Sears, W.R. (1947). On projectiles of minimum wave drag. 
      Quarterly of Applied Mathematics, 4(4), 361–366.
  [6] Fay & Riddell (1958). Theory of stagnation point heat transfer.
      J. Aero. Sci. 25(2), 73–85.
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from core.atmosphere import ISAModel
from core.aerodynamics import AeroAnalyzer, TrajectoryAnalyzer
from core.shock_wave import oblique_shock, prandtl_meyer_expansion, normal_shock
from geometry.shapes import (make_generic_rocket, make_falcon9,
                               make_optimized_rocket, NoseType, VehicleGeometry)


# ─── 색상 팔레트 ──────────────────────────────────────────────────────────────
COLORS = {
    'generic':   '#2196F3',   # 파란색
    'falcon9':   '#FF5722',   # 주황색
    'optimized': '#4CAF50',   # 초록색
    'shock':     '#F44336',   # 빨강 (충격파)
    'expansion': '#00BCD4',   # 청록 (팽창파)
    'drag':      '#9C27B0',   # 보라 (항력)
}

PHASE_COLORS = {
    '발사 준비': '#607D8B',
    '아음속':   '#2196F3',
    '천음속':   '#FF9800',
    'MaxQ':     '#F44336',
    '초음속':   '#9C27B0',
    '극초음속': '#795548',
}


def run_full_analysis():
    """전체 3종 발사체 비교 분석"""
    print("=" * 70)
    print("   우주발사체 공기역학 시뮬레이션 Suite")
    print("   Launch Vehicle Aerodynamic Analysis Suite")
    print("   ISA 대기 모델 기반 | 충격파·팽창파·항력 해석")
    print("=" * 70)

    isa = ISAModel()

    # 발사체 생성
    vehicles = {
        "Generic LV":  make_generic_rocket(),
        "Falcon 9":    make_falcon9(),
        "Optimized":   make_optimized_rocket(),
    }

    # 마하수 스윕 범위 (0.1 ~ 10)
    mach_array = np.concatenate([
        np.linspace(0.1, 0.79, 20),   # 아음속
        np.linspace(0.80, 1.20, 25),  # 천음속 (세밀하게)
        np.linspace(1.21, 3.0, 30),   # 초음속
        np.linspace(3.1, 10.0, 20),   # 극초음속
    ])

    results = {}
    for name, geo in vehicles.items():
        print(f"\n[분석 중] {geo.name}")
        print(geo.summary())
        ta = TrajectoryAnalyzer(geo, isa)
        results[name] = ta.mach_sweep(mach_array)

    # 시각화
    _plot_cd_vs_mach(results, mach_array)
    _plot_shock_features(vehicles["Falcon 9"], isa)
    _plot_surface_distribution(vehicles, isa, mach=2.0, altitude=30000)
    _plot_geometry_comparison(vehicles)
    _plot_phase_analysis(results, mach_array)

    print("\n[완료] 그래프가 'results/' 폴더에 저장되었습니다.")
    return results


def _plot_cd_vs_mach(results: dict, mach_array: np.ndarray):
    """항력계수 vs 마하수 그래프"""
    Path("results").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("항력계수 해석 | CD vs Mach Number", fontsize=14, fontweight='bold')

    names = list(results.keys())
    color_list = [COLORS['generic'], COLORS['falcon9'], COLORS['optimized']]

    ax = axes[0, 0]
    for name, color in zip(names, color_list):
        Cd = [r.Cd_total for r in results[name]]
        ax.plot(mach_array[:len(Cd)], Cd, color=color, lw=2, label=name)
    ax.axvspan(0.8, 1.2, alpha=0.1, color='orange', label='천음속 영역')
    ax.axvline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel("마하수 M"); ax.set_ylabel("CD (전체)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("전체 항력계수 (Total Drag)")

    ax = axes[0, 1]
    for name, color in zip(names, color_list):
        Cd_w = [r.Cd_wave for r in results[name]]
        ax.plot(mach_array[:len(Cd_w)], Cd_w, color=color, lw=2, label=name)
    ax.set_xlabel("마하수 M"); ax.set_ylabel("CD_wave"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("파동 항력계수 (Wave Drag)")

    ax = axes[1, 0]
    for name, color in zip(names, color_list):
        Cd_f = [r.Cd_friction for r in results[name]]
        ax.plot(mach_array[:len(Cd_f)], Cd_f, color=color, lw=2, label=name)
    ax.set_xlabel("마하수 M"); ax.set_ylabel("CD_friction"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("마찰 항력계수 (Skin Friction)")

    ax = axes[1, 1]
    for name, color in zip(names, color_list):
        q_list = [r.dynamic_pressure / 1000 for r in results[name]]
        ax.plot(mach_array[:len(q_list)], q_list, color=color, lw=2, label=name)
    ax.axhline(85, color='red', ls='--', lw=1, label='Falcon 9 MaxQ ~85 kPa')
    ax.set_xlabel("마하수 M"); ax.set_ylabel("동압 q [kPa]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("동압 (Dynamic Pressure)")

    plt.tight_layout()
    plt.savefig("results/cd_vs_mach.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: results/cd_vs_mach.png")


def _plot_shock_features(geo: VehicleGeometry, isa: ISAModel):
    """충격파 구조 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"충격파 구조 | {geo.name}", fontsize=12, fontweight='bold')

    mach_cases = [1.5, 3.0, 7.0]
    altitudes  = [15000, 30000, 60000]

    for ax, M, h in zip(axes, mach_cases, altitudes):
        atm = isa.get_state(h)
        analyzer = AeroAnalyzer(geo, isa)
        res = analyzer.analyze(M, h)

        # 발사체 형상 그리기
        ax.fill_betweenx(geo.x_profile, -geo.r_profile, geo.r_profile,
                          color='#37474F', alpha=0.8)

        # 충격파 / 팽창파 그리기
        for feat in res.shock_features:
            x0 = feat.x_location
            if feat.feature_type in ("oblique_shock", "normal_shock"):
                length = geo.total_length * 0.4
                angle_rad = np.radians(feat.angle if feat.angle < 90 else 90)
                dx = length * np.cos(angle_rad) * 0.3
                r_at_x = np.interp(x0, geo.x_profile, geo.r_profile)
                # 충격파 선 (양쪽)
                for sign in [1, -1]:
                    ax.plot([r_at_x * sign, (r_at_x + dx) * sign],
                            [x0, x0 + length * 0.5],
                            color=COLORS['shock'], lw=2, alpha=0.85)
            elif feat.feature_type == "expansion_fan":
                r_at_x = np.interp(x0, geo.x_profile, geo.r_profile)
                for angle in np.linspace(0, 30, 5):
                    rad = np.radians(angle)
                    length = geo.total_length * 0.25
                    for sign in [1, -1]:
                        ax.plot([r_at_x * sign, (r_at_x + length * np.sin(rad)) * sign],
                                [x0, x0 + length * np.cos(rad)],
                                color=COLORS['expansion'], lw=1, alpha=0.6)

        ax.set_xlim(-geo.max_diameter, geo.max_diameter)
        ax.set_xlabel("반경 [m]")
        ax.set_ylabel("축방향 [m]")
        ax.set_title(f"M = {M} | h = {h/1000:.0f} km\n"
                     f"CD = {res.Cd_total:.4f}")
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("results/shock_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: results/shock_structure.png")


def _plot_surface_distribution(vehicles: dict, isa: ISAModel,
                                mach: float, altitude: float):
    """표면 압력/온도 분포"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"표면 분포 | M={mach}, h={altitude/1000:.0f} km", fontsize=12)

    for col, (name, geo) in enumerate(vehicles.items()):
        analyzer = AeroAnalyzer(geo, isa)
        res = analyzer.analyze(mach, altitude)

        ax_p = axes[0, col]
        ax_T = axes[1, col]

        ax_p.plot(res.x_surface, res.Cp_surface, color=COLORS['drag'], lw=2)
        ax_p.fill_between(res.x_surface, res.Cp_surface, alpha=0.3, color=COLORS['drag'])
        ax_p.axhline(0, color='black', lw=0.5)
        ax_p.set_title(f"{name}\nCp 분포")
        ax_p.set_xlabel("x [m]"); ax_p.set_ylabel("Cp")
        ax_p.grid(alpha=0.3)

        ax_T.plot(res.x_surface, res.T_surface, color=COLORS['shock'], lw=2)
        ax_T.fill_between(res.x_surface, res.T_surface, alpha=0.3, color=COLORS['shock'])
        ax_T.set_title(f"온도 분포 [K]")
        ax_T.set_xlabel("x [m]"); ax_T.set_ylabel("T [K]")
        ax_T.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/surface_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: results/surface_distribution.png")


def _plot_geometry_comparison(vehicles: dict):
    """발사체 형상 비교"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle("발사체 형상 비교 | Geometry Comparison", fontsize=12, fontweight='bold')

    color_list = [COLORS['generic'], COLORS['falcon9'], COLORS['optimized']]
    for ax, (name, geo), color in zip(axes, vehicles.items(), color_list):
        x = geo.x_profile
        r = geo.r_profile
        ax.fill_betweenx(x, -r, r, color=color, alpha=0.7, label=name)
        ax.plot(r, x, color=color, lw=2)
        ax.plot(-r, x, color=color, lw=2)
        ax.set_xlim(-geo.max_diameter * 1.5, geo.max_diameter * 1.5)
        ax.set_xlabel("반경 [m]"); ax.set_ylabel("축방향 [m]")
        ax.set_title(f"{name}\nL={geo.total_length:.1f}m, L/D={geo.fineness_ratio:.1f}")
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("results/geometry_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: results/geometry_comparison.png")


def _plot_phase_analysis(results: dict, mach_array: np.ndarray):
    """비행 단계별 분석"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("비행 단계별 전체 항력계수 비교", fontsize=12, fontweight='bold')

    names = list(results.keys())
    color_list = [COLORS['generic'], COLORS['falcon9'], COLORS['optimized']]

    phase_boundaries = [
        (0.0, 0.8, '#E3F2FD', '아음속'),
        (0.8, 1.2, '#FFF3E0', '천음속'),
        (1.2, 5.0, '#F3E5F5', '초음속'),
        (5.0, 10.5, '#EFEBE9', '극초음속'),
    ]
    for x0, x1, c, label in phase_boundaries:
        ax.axvspan(x0, x1, alpha=0.5, color=c, label=label)

    for name, color in zip(names, color_list):
        Cd = [r.Cd_total for r in results[name]]
        ax.plot(mach_array[:len(Cd)], Cd, color=color, lw=2.5, label=name)

    ax.axvline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel("마하수 M", fontsize=11)
    ax.set_ylabel("CD (전체 항력계수)", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10.5)

    plt.tight_layout()
    plt.savefig("results/phase_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: results/phase_analysis.png")


def analyze_single(vehicle_name: str, mach: float, altitude: float):
    """단일 조건 분석"""
    isa = ISAModel()
    if vehicle_name == "falcon9":
        geo = make_falcon9()
    elif vehicle_name == "optimized":
        geo = make_optimized_rocket()
    else:
        geo = make_generic_rocket()

    analyzer = AeroAnalyzer(geo, isa)
    res = analyzer.analyze(mach, altitude)
    print(res)

    print("\n  충격파/팽창파 특성:")
    for feat in res.shock_features:
        print(f"    [{feat.feature_type}] x={feat.x_location:.1f}m, "
              f"M1={feat.M_upstream:.2f}→M2={feat.M_downstream:.2f}, "
              f"p2/p1={feat.pressure_ratio:.3f}")
    return res


def load_and_analyze_stl(filepath: str, scale: float, mach: float, altitude: float):
    """STL 파일 로드 및 분석"""
    from geometry.stl_loader import load_stl, STLGeometry
    import types

    stl_geo = load_stl(filepath, scale=scale)
    print(stl_geo.summary())

    # STLGeometry → VehicleGeometry 호환 래퍼
    class STLVehicleWrapper(VehicleGeometry):
        def __init__(self, stl: STLGeometry):
            self.name = stl.name
            self.sections = []
            self.fins = None
            self.x_profile = stl.x_profile
            self.r_profile = stl.r_profile
            self.total_length = stl.total_length
            self._frontal_area = stl.frontal_area
            self._wetted_area  = stl.wetted_area

        @property
        def frontal_area(self): return self._frontal_area
        @property
        def wetted_area(self): return self._wetted_area
        @property
        def base_area(self):
            return np.pi * (self.r_profile[-1]) ** 2
        @property
        def max_diameter(self):
            return 2 * np.max(self.r_profile)
        @property
        def fineness_ratio(self):
            return self.total_length / self.max_diameter

    wrapper = STLVehicleWrapper(stl_geo)
    isa = ISAModel()
    analyzer = AeroAnalyzer(wrapper, isa)
    res = analyzer.analyze(mach, altitude)
    print(res)
    return res


def main():
    parser = argparse.ArgumentParser(
        description="우주발사체 공기역학 시뮬레이션 Suite")
    parser.add_argument('--vehicle', default='all',
                        choices=['all', 'generic', 'falcon9', 'optimized'],
                        help="분석할 발사체 선택")
    parser.add_argument('--mach', type=float, default=None,
                        help="마하수 (단일 조건 분석 시)")
    parser.add_argument('--alt', type=float, default=None,
                        help="고도 [m] (단일 조건 분석 시)")
    parser.add_argument('--stl', type=str, default=None,
                        help="STL 파일 경로")
    parser.add_argument('--scale', type=float, default=1.0,
                        help="STL 단위 스케일 (예: 0.001 = mm→m)")
    args = parser.parse_args()

    if args.stl:
        M = args.mach or 2.0
        h = args.alt  or 30000
        load_and_analyze_stl(args.stl, args.scale, M, h)
    elif args.mach is not None or args.alt is not None:
        M = args.mach or 2.0
        h = args.alt  or 30000
        analyze_single(args.vehicle if args.vehicle != 'all' else 'falcon9', M, h)
    else:
        run_full_analysis()


if __name__ == "__main__":
    main()
