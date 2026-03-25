# 우주발사체 공기역학 시뮬레이션 Suite
## Launch Vehicle Aerodynamic Analysis Suite

### 구성 파일 Structure
- `main.py`           : 메인 실행 파일 (Main runner)
- `core/atmosphere.py`: ISA 대기 모델 (ISA Atmosphere model)
- `core/aerodynamics.py`: 공기역학 해석 엔진 (Aero analysis engine)
- `core/shock_wave.py`: 충격파/팽창파 계산 (Shock/Expansion wave)
- `geometry/shapes.py`: 발사체 형상 정의 (Vehicle geometry)
- `geometry/stl_loader.py`: STL 파일 로더 (STL file loader)
- `visualization/plotter.py`: 시각화 모듈 (Visualization)
- `optimization/optimizer.py`: 형상 최적화 (Shape optimization)

### 참고문헌 References
- Anderson, J.D. (2003). Modern Compressible Flow. McGraw-Hill.
- Hoerner, S.F. (1965). Fluid-Dynamic Drag. Published by the author.
- DATCOM: USAF Stability and Control Datcom (1978)
- NASA TM-2011-216962: Rocket Aerodynamics
- ISA: ISO 2533:1975 Standard Atmosphere
