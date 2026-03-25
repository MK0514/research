"""
STL 파일 로더 (STL File Loader)
발사체 형상을 STL 파일에서 읽어 공기역학 해석용 형상으로 변환

References:
  - ISO/IEC 19776 (STL format specification)
  - numpy-stl library documentation
  - Geuzaine, C. & Remacle, J.F. (2009). "Gmsh: A 3D finite element mesh
    generator." Int. J. Numer. Methods Eng. 79(11):1309–1331.
"""

import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from stl import mesh as stl_mesh
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("[경고] numpy-stl 패키지가 설치되지 않았습니다.")
    print("       pip install numpy-stl 로 설치하세요.")


@dataclass
class STLGeometry:
    """STL 파일에서 읽은 발사체 형상"""
    name: str
    vertices: np.ndarray    # (N,3) float array [m]
    normals: np.ndarray     # (N,3) surface normals
    triangles: np.ndarray   # (N,3,3) triangle vertices

    # 공기역학 해석용 파생 변수
    x_profile: np.ndarray = None   # 축방향 좌표 [m]
    r_profile: np.ndarray = None   # 반경 [m]
    total_length: float = 0.0
    max_diameter: float = 0.0
    base_diameter: float = 0.0
    frontal_area: float = 0.0
    wetted_area: float = 0.0

    def summary(self) -> str:
        return (
            f"=== STL 형상: {self.name} ===\n"
            f"  삼각형 수     : {len(self.triangles):,}\n"
            f"  전체 길이     : {self.total_length:.3f} m\n"
            f"  최대 직경     : {self.max_diameter:.3f} m\n"
            f"  기저 직경     : {self.base_diameter:.3f} m\n"
            f"  세장비 (L/D)  : {(self.total_length / self.max_diameter):.2f}\n"
            f"  습윤 면적     : {self.wetted_area:.3f} m²\n"
        )


def load_stl(filepath: str,
             scale: float = 1.0,
             axis_direction: str = "x",
             n_profile_points: int = 200) -> STLGeometry:
    """
    STL 파일을 읽어 발사체 형상으로 변환
    Load STL file and convert to aerodynamic geometry.
    
    Args:
        filepath         : STL 파일 경로 (binary or ASCII)
        scale            : 단위 변환 스케일 (mm→m: scale=0.001)
        axis_direction   : 발사 방향 축 ('x', 'y', 'z')
        n_profile_points : 단면 프로파일 이산화 수

    Returns:
        STLGeometry 객체
    
    STL 형식 참고:
      - Binary STL: 80B header + uint32 triangle count + N*(12B normal + 36B verts + 2B attr)
      - ASCII STL: "solid name ... endsolid name"
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {filepath}")

    print(f"[STL] '{filepath.name}' 로드 중...")

    if STL_AVAILABLE:
        geo = _load_with_numpy_stl(filepath, scale, axis_direction, n_profile_points)
    else:
        geo = _load_stl_manual(filepath, scale, axis_direction, n_profile_points)

    print(f"[STL] 로드 완료: {len(geo.triangles):,}개 삼각형")
    return geo


def _load_with_numpy_stl(filepath: Path, scale: float,
                          axis_direction: str,
                          n_pts: int) -> STLGeometry:
    """numpy-stl을 사용한 STL 로드"""
    from stl import mesh as stl_mesh
    m = stl_mesh.Mesh.from_file(str(filepath))

    verts    = m.vectors * scale       # (N,3,3)
    normals  = m.normals               # (N,3)
    triangles = verts

    all_verts = verts.reshape(-1, 3)
    return _build_geometry(filepath.stem, all_verts, normals, triangles,
                           axis_direction, n_pts)


def _load_stl_manual(filepath: Path, scale: float,
                      axis_direction: str, n_pts: int) -> STLGeometry:
    """
    외부 라이브러리 없이 STL 파싱 (Manual STL parser)
    Binary STL 및 ASCII STL 모두 지원
    """
    with open(filepath, 'rb') as f:
        header = f.read(80)
        is_ascii = header[:5].lower() == b'solid'

    if is_ascii:
        triangles, normals = _parse_ascii_stl(filepath, scale)
    else:
        triangles, normals = _parse_binary_stl(filepath, scale)

    all_verts = triangles.reshape(-1, 3)
    return _build_geometry(filepath.stem, all_verts, normals, triangles,
                           axis_direction, n_pts)


def _parse_binary_stl(filepath: Path, scale: float):
    """Binary STL 파서"""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        triangles = np.zeros((n_tri, 3, 3), dtype=np.float32)
        normals   = np.zeros((n_tri, 3),    dtype=np.float32)

        for i in range(n_tri):
            normals[i] = struct.unpack('<3f', f.read(12))
            for j in range(3):
                triangles[i, j] = struct.unpack('<3f', f.read(12))
            f.read(2)  # attribute byte count

    return triangles * scale, normals


def _parse_ascii_stl(filepath: Path, scale: float):
    """ASCII STL 파서"""
    triangles_list, normals_list = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith('facet normal'):
            parts = lines[i].split()
            n = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
            normals_list.append(n)
            verts = []
            i += 2  # skip 'outer loop'
            for _ in range(3):
                i += 1
                p = lines[i].split()
                verts.append([float(p[-3]), float(p[-2]), float(p[-1])])
            triangles_list.append(verts)
        i += 1

    triangles = np.array(triangles_list, dtype=np.float32) * scale
    normals   = np.array(normals_list,   dtype=np.float32)
    return triangles, normals


def _build_geometry(name: str, all_verts: np.ndarray,
                     normals: np.ndarray, triangles: np.ndarray,
                     axis_direction: str, n_pts: int) -> STLGeometry:
    """
    꼭짓점 데이터로부터 공기역학 해석용 프로파일 생성
    Build aerodynamic profile from vertex data.
    
    회전체 가정: 지정 축 방향이 비행 방향
    Assumes body of revolution along the specified axis.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map.get(axis_direction.lower(), 0)
    rad_axes = [i for i in range(3) if i != ax]

    # 각 꼭짓점의 반경 계산 (Radial distance from axis)
    x_vals = all_verts[:, ax]
    r_vals = np.sqrt(all_verts[:, rad_axes[0]] ** 2 +
                     all_verts[:, rad_axes[1]] ** 2)

    x_min, x_max = x_vals.min(), x_vals.max()
    total_length = x_max - x_min

    # 단면별 최대 반경 프로파일 생성
    # (Profile: max radius at each x-station)
    x_stations = np.linspace(x_min, x_max, n_pts)
    r_profile  = np.zeros(n_pts)
    bin_edges  = np.linspace(x_min, x_max, n_pts + 1)

    for i in range(n_pts):
        mask = (x_vals >= bin_edges[i]) & (x_vals < bin_edges[i + 1])
        if mask.any():
            r_profile[i] = r_vals[mask].max()
        elif i > 0:
            r_profile[i] = r_profile[i - 1]

    # 습윤 면적 계산
    dx  = np.diff(x_stations)
    dr  = np.diff(r_profile)
    ds  = np.sqrt(dx ** 2 + dr ** 2)
    r_m = (r_profile[:-1] + r_profile[1:]) / 2
    wetted_area = float(np.sum(2 * np.pi * r_m * ds))

    max_r = r_profile.max()

    geo = STLGeometry(
        name=name,
        vertices=all_verts,
        normals=normals,
        triangles=triangles,
        x_profile=x_stations - x_min,
        r_profile=r_profile,
        total_length=total_length,
        max_diameter=2 * max_r,
        base_diameter=2 * r_profile[-1],
        frontal_area=np.pi * max_r ** 2,
        wetted_area=wetted_area,
    )
    return geo


def check_stl_dependencies():
    """STL 로더 의존성 확인"""
    print("=== STL 로더 의존성 확인 ===")
    print(f"  numpy-stl : {'설치됨 (권장)' if STL_AVAILABLE else '미설치 (수동 파서 사용)'}")
    print(f"  numpy     : 설치됨")
    if not STL_AVAILABLE:
        print("\n  [권장] pip install numpy-stl")


if __name__ == "__main__":
    check_stl_dependencies()
    print("\n사용 예시:")
    print("  from geometry.stl_loader import load_stl")
    print("  geo = load_stl('my_rocket.stl', scale=0.001)  # mm -> m")
    print("  print(geo.summary())")
