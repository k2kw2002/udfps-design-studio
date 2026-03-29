"""
UDFPS COE Design Studio — Streamlit Prototype
기술보고서 5.8절 Design Studio UI 사양 기반
6개 탭: 아키텍처 | 설계(워크플로우) | 데이터 흐름 | 핵심 수식 | Pareto 탐색기 | 로드맵
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json, time, os

# ── 페이지 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title='UDFPS COE Design Studio',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 한글 폰트 설정 (Streamlit Cloud 호환)
import platform, os as _os
if platform.system() != 'Windows':
    _font_dir = _os.path.expanduser('~/.fonts')
    _font_path = _os.path.join(_font_dir, 'NotoSansKR.ttf')
    if not _os.path.exists(_font_path):
        _os.makedirs(_font_dir, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf',
            _font_path
        )
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = ['Noto Sans KR', 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 물리 함수 ─────────────────────────────────────────────────────────────
WAVELENGTH_UM = 0.85
K0_UM = 2 * np.pi / WAVELENGTH_UM
N_SiO2, N_TiO2 = 1.46, 2.35
N_GLASS = 1.52

# DX (4층) / DX+ (8층) 코팅 구조
COATING_PRESETS = {
    'Gorilla DX (4층)': {
        'thick_nm': [49.5, 19.9, 29.6, 130.4],
        'ns': [N_SiO2, N_TiO2, N_SiO2, N_TiO2],
        'mats': ['SiO2', 'TiO2', 'SiO2', 'TiO2'],
    },
    'Gorilla DX+ (8층)': {
        'thick_nm': [38.1, 30.1, 43.5, 63.7, 0.8, 25.6, 8.1, 190.8],
        'ns': [N_SiO2, N_TiO2, N_SiO2, N_TiO2, N_SiO2, N_TiO2, N_SiO2, N_TiO2],
        'mats': ['SiO2', 'TiO2', 'SiO2', 'TiO2', 'SiO2', 'TiO2', 'SiO2', 'TiO2'],
    },
}

BASE_THICK_UM = np.array([49.5, 19.9, 29.6, 130.4]) * 1e-3
LAYER_NS = [N_SiO2, N_TiO2, N_SiO2, N_TiO2]
N_FIELD = 512
X_UM = np.linspace(-300.0, 300.0, N_FIELD, dtype=np.float64)
DX_UM = float(X_UM[1] - X_UM[0])
SIGMA0 = 100.0


def tmm_phase_general(thick_nm_arr, ns_arr, d_scales, theta_deg):
    """일반화된 TMM: 임의 층 수 지원"""
    lam_m = WAVELENGTH_UM * 1e-6
    k0m = 2 * np.pi / lam_m
    n_in, n_out = N_GLASS, 1.00
    th = np.radians(theta_deg)
    d_m = np.array(thick_nm_arr) * 1e-9 * d_scales
    cos_list = [np.cos(th + 0j)]
    for nl in ns_arr:
        sn = n_in * np.sin(th) / nl
        cos_list.append(np.sqrt(1 - sn**2 + 0j))
    sn_o = n_in * np.sin(th) / n_out
    cos_o = np.sqrt(1 - sn_o**2 + 0j)
    M = np.eye(2, dtype=complex)
    for i, (nl, d) in enumerate(zip(ns_arr, d_m)):
        phi = k0m * nl * cos_list[i + 1] * d
        m = np.array([
            [np.cos(phi), -1j * np.sin(phi) / (nl * cos_list[i + 1])],
            [-1j * nl * cos_list[i + 1] * np.sin(phi), np.cos(phi)]
        ], dtype=complex)
        M = M @ m
    p_i = n_in * cos_list[0]
    p_o = n_out * cos_o
    t = 2 * p_i / ((M[0, 0] + M[0, 1] * p_o) * p_i + (M[1, 0] + M[1, 1] * p_o) + 1e-30)
    T = float(np.real(p_o / p_i * np.abs(t)**2))
    return max(0.0, min(1.0, T)), float(np.angle(t))


def tmm_phase(d_scales, theta_deg):
    """DX 4층 호환 래퍼"""
    return tmm_phase_general([49.5, 19.9, 29.6, 130.4],
                              LAYER_NS, d_scales, theta_deg)


def asm_1d(U_in, dx, d_um, n=1.0):
    kx = 2 * np.pi * np.fft.fftfreq(len(U_in), d=dx)
    kz2 = (n * K0_UM)**2 - kx**2
    kz = np.sqrt(np.where(kz2 >= 0, kz2, 0) + 0j)
    return np.fft.ifft(np.fft.fft(U_in) * np.exp(1j * kz * d_um))


def full_pipeline(d_scales, delta_bm, theta_deg,
                   w1=30.0, w2=40.0, d_cg=550.0, d_bm=30.0,
                   coating_preset=None):
    if coating_preset and coating_preset in COATING_PRESETS:
        cp = COATING_PRESETS[coating_preset]
        T_c, phi = tmm_phase_general(cp['thick_nm'], cp['ns'], d_scales, theta_deg)
        _, phi0 = tmm_phase_general(cp['thick_nm'], cp['ns'], d_scales, 0.0)
    else:
        T_c, phi = tmm_phase(d_scales, theta_deg)
        _, phi0 = tmm_phase(d_scales, 0.0)
    dphi = phi - phi0
    kx_t = K0_UM * np.sin(np.radians(theta_deg))

    U_in = (np.exp(-X_UM**2 / (2 * SIGMA0**2)) *
            np.exp(1j * (kx_t * X_UM + dphi))).astype(np.complex128)

    # ASM: Cover Glass
    U_cg = asm_1d(U_in, DX_UM, d_cg, n=N_GLASS)
    # BM2 mask
    U_cg = U_cg * (np.abs(X_UM) <= w2 / 2)
    # ASM: BM gap
    U_bm = asm_1d(U_cg, DX_UM, d_bm)
    # BM1 mask
    U_out = U_bm * (np.abs(X_UM - delta_bm) <= w1 / 2)

    I = np.abs(U_out)**2 * T_c
    # metrics
    roi = np.abs(X_UM) <= 200.0
    xr, Ir = X_UM[roi], I[roi]
    T_total = float(np.sum(I) * DX_UM)
    # MTF
    c_ = I[np.abs(X_UM) < 15.0].mean() if np.any(np.abs(X_UM) < 15.0) else 0
    sl_ = I[(np.abs(X_UM) > 25.0) & (np.abs(X_UM) < 80.0)].mean() + 1e-20
    MTF = float(c_ / (c_ + sl_))
    # skewness
    if Ir.sum() > 1e-20:
        In = Ir / Ir.sum()
        mu = (xr * In).sum()
        s2 = ((xr - mu)**2 * In).sum()
        skew = float(((xr - mu)**3 * In).sum() / (s2**1.5 + 1e-30))
    else:
        skew = 0.0

    return {
        'MTF': MTF, 'T_total': T_total, 'skewness': skew,
        'T_coating': T_c, 'dphi_deg': np.degrees(dphi),
        'PSF': I, 'U_in': np.abs(U_in)**2
    }


# ── 사이드바 ──────────────────────────────────────────────────────────────
st.sidebar.title('UDFPS COE Design Studio')
st.sidebar.caption('Physics-Informed AI 설계 플랫폼 v0.2')
st.sidebar.divider()

# 설계 이력 저장소 초기화
if 'design_history' not in st.session_state:
    st.session_state['design_history'] = []

# ── 탭 구성 (8개) ────────────────────────────────────────────────────────
tab1, tab2, tab7, tab8, tab3, tab4, tab5, tab10, tab6, tab9 = st.tabs([
    '1. 아키텍처', '2. 설계',
    '3. 다각도 Robustness', '4. 민감도 분석',
    '5. 데이터 흐름', '6. 핵심 수식',
    '7. Pareto 탐색기', '8. 지문 시뮬레이션',
    '9. 로드맵', '10. 설계 이력'
])

# ══════════════════════════════════════════════════════════════════════════
# 탭 1: 아키텍처
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header('6계층 플랫폼 아키텍처')
    st.markdown('> 기술보고서 5.2절 — TMM·ASM·PINN·FNO·BoTorch 연결 구조')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('플랫폼 계층 구조')
        layers_data = [
            ('L6', 'Design Studio UI', 'FastAPI + WebSocket', '#E3F2FD'),
            ('L5', '데이터 파이프라인', 'MLflow + DVC + Celery', '#F3E5F5'),
            ('L4', '최적화 엔진', 'BoTorch qNEHVI 3목적', '#FFF3E0'),
            ('L3', 'FNO Surrogate', 'PINN 증류 → <1ms 추론', '#E8F5E9'),
            ('L2', 'Helmholtz PINN', 'SIREN + Fourier Feature', '#FFEBEE'),
            ('L1', '물리 엔진 ★', 'TMM + ASM + (LightTools)', '#E0F7FA'),
        ]

        for lid, name, desc, color in layers_data:
            st.markdown(
                f'<div style="background:{color}; padding:12px; '
                f'border-radius:8px; margin:4px 0; border-left:4px solid #333;">'
                f'<b>{lid}</b> &nbsp; {name}<br/>'
                f'<small style="color:#666;">{desc}</small></div>',
                unsafe_allow_html=True
            )

    with col2:
        st.subheader('UDFPS COE 스택 구조')
        stack = [
            ('Finger', '#FFDAB9'),
            ('AR Coating (DX/DX+)', '#90EE90'),
            ('Cover Glass 550μm', '#87CEEB'),
            ('Interlayer', '#F5F5DC'),
            ('BM2 (w2=40μm)', '#696969'),
            ('Interlayer (d_int)', '#F5F5DC'),
            ('BM1 (w1=30μm)', '#696969'),
            ('Encapsulation', '#DDA0DD'),
            ('Sensor (PD)', '#FFD700'),
        ]

        for name, color in stack:
            tc = 'white' if color == '#696969' else 'black'
            st.markdown(
                f'<div style="background:{color}; color:{tc}; padding:8px; '
                f'text-align:center; border-radius:4px; margin:2px 0; '
                f'font-weight:bold; font-size:14px;">{name}</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader('핵심 차별점 3가지')
    c1, c2, c3 = st.columns(3)
    c1.metric('혁신 1', 'PSF Skewness', '목적함수에 포함')
    c2.metric('혁신 2', 'L_phase Loss', 'TMM→PINN 위상 주입')
    c3.metric('혁신 3', '도메인 축소', '573μm → 30μm (20×)')


# ══════════════════════════════════════════════════════════════════════════
# 탭 2: 설계 (워크플로우)
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header('실시간 광학 설계')
    st.markdown('> 설계 변수 조절 → TMM → ASM(CG) → BM2 → ASM(BM) → BM1 → PSF')

    # 설계 변수 입력
    with st.sidebar:
        st.subheader('코팅 구조')
        coating_type = st.selectbox('AR 코팅 선택',
            list(COATING_PRESETS.keys()), key='coating_sel')
        cp = COATING_PRESETS[coating_type]
        n_layers = len(cp['thick_nm'])

        st.subheader('층 두께 배율')
        d_sliders = []
        for i in range(n_layers):
            val = st.slider(f'd{i+1} ({cp["mats"][i]}, {cp["thick_nm"][i]:.1f}nm)',
                           0.5, 1.5, 1.0, 0.05, key=f'ds_{i}')
            d_sliders.append(val)

        st.caption('BM 구조')
        delta_bm = st.slider('delta_BM (μm)', -30.0, 30.0, 0.0, 1.0, key='dbm')
        w1 = st.slider('w1 - BM1 아퍼처 (μm)', 10.0, 60.0, 30.0, 1.0)
        w2 = st.slider('w2 - BM2 아퍼처 (μm)', 10.0, 60.0, 40.0, 1.0)

        st.caption('입사 조건')
        theta = st.slider('입사각 θ (°)', 0, 40, 30, 1)

        st.divider()

        # ── 설계 저장 ──
        if st.button('현재 설계 저장', use_container_width=True):
            entry = {
                'time': time.strftime('%H:%M:%S'),
                'coating': coating_type,
                'd_scales': [round(v, 3) for v in d_sliders],
                'delta_bm': delta_bm, 'theta': theta,
                'w1': w1, 'w2': w2,
            }
            # 결과는 아래에서 추가
            st.session_state['_save_pending'] = entry

        st.divider()

        # ── 자동 최적화 버튼 ──
        st.subheader('자동 최적화')
        opt_target = st.selectbox('최적화 목표',
            ['skewness 최소화', 'MTF 최대화', 'T_total 최대화', '종합 최적 (가중합)'])
        run_opt = st.button('최적 설계 자동 탐색', type='primary', use_container_width=True)

    d_scales = np.array(d_sliders)

    # ── 자동 최적화 실행 ──
    if run_opt:
        with st.spinner('최적 설계 탐색 중...'):
            from scipy.optimize import differential_evolution

            def objective(params):
                ds = np.array(params[:n_layers])
                db = params[n_layers]
                r = full_pipeline(ds, db, theta, w1=w1, w2=w2,
                                  coating_preset=coating_type)
                if opt_target == 'skewness 최소화':
                    return abs(r['skewness'])
                elif opt_target == 'MTF 최대화':
                    return -r['MTF']
                elif opt_target == 'T_total 최대화':
                    return -r['T_total']
                else:  # 종합
                    return -(r['MTF'] * 0.3 + r['T_total'] * 50 * 0.3
                             + (1 - abs(r['skewness'])) * 0.4)

            bounds_opt = [(0.5, 1.5)] * n_layers + [(-30, 30)]
            opt_result = differential_evolution(
                objective, bounds_opt, seed=42,
                maxiter=30, popsize=10, tol=1e-4,
                mutation=(0.5, 1.5), recombination=0.8
            )

            best_ds = np.array(opt_result.x[:n_layers])
            best_db = opt_result.x[n_layers]
            best_r = full_pipeline(best_ds, best_db, theta, w1=w1, w2=w2,
                                   coating_preset=coating_type)
            st.session_state['auto_opt'] = {
                'd_scales': best_ds, 'delta_bm': best_db,
                'result': best_r, 'target': opt_target
            }

    # ── 자동 최적화 결과 표시 ──
    if 'auto_opt' in st.session_state:
        ao = st.session_state['auto_opt']
        st.success(f'최적 설계 발견! (목표: {ao["target"]})')

        co1, co2, co3, co4, co5 = st.columns(5)
        co1.metric('MTF', f'{ao["result"]["MTF"]:.4f}')
        co2.metric('T_total', f'{ao["result"]["T_total"]:.5f}')
        co3.metric('|skew|', f'{abs(ao["result"]["skewness"]):.4f}')
        co4.metric('delta_BM', f'{ao["delta_bm"]:.1f} μm')
        co5.metric('T 코팅', f'{ao["result"]["T_coating"]*100:.1f}%')

        thick_opt = BASE_THICK_UM * ao['d_scales'] * 1e3
        st.markdown(
            f'**최적 코팅 두께:** '
            f'SiO2({thick_opt[0]:.1f}nm) / TiO2({thick_opt[1]:.1f}nm) / '
            f'SiO2({thick_opt[2]:.1f}nm) / TiO2({thick_opt[3]:.1f}nm)'
        )
        st.caption('이 값을 사이드바 슬라이더에 적용하면 PSF를 직접 확인할 수 있습니다.')
        st.divider()

    # 시뮬레이션 실행
    with st.spinner('파이프라인 계산 중...'):
        result = full_pipeline(d_scales, delta_bm, theta, w1=w1, w2=w2,
                               coating_preset=coating_type)
        result_base = full_pipeline(np.ones(n_layers), 0.0, theta, w1=w1, w2=w2,
                                    coating_preset=coating_type)

    # 설계 저장 처리
    if '_save_pending' in st.session_state:
        entry = st.session_state.pop('_save_pending')
        entry['MTF'] = round(result['MTF'], 4)
        entry['T_total'] = round(result['T_total'], 6)
        entry['skewness'] = round(result['skewness'], 4)
        st.session_state['design_history'].append(entry)
        st.toast('설계가 저장되었습니다!')

    # 성능 지표
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric('MTF@ridge', f'{result["MTF"]:.4f}',
                  f'{(result["MTF"]-result_base["MTF"])*100:+.2f}%')
    col_m2.metric('T_total', f'{result["T_total"]:.5f}',
                  f'{(result["T_total"]-result_base["T_total"])/max(result_base["T_total"],1e-10)*100:+.1f}%')
    col_m3.metric('|skewness|', f'{abs(result["skewness"]):.4f}',
                  f'{abs(result["skewness"])-abs(result_base["skewness"]):.4f}',
                  delta_color='inverse')
    col_m4.metric('TMM Δφ(θ)', f'{result["dphi_deg"]:.2f}°',
                  f'T={result["T_coating"]*100:.1f}%')

    # 시각화
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        psf_norm = result['PSF'] / (result['PSF'].max() + 1e-20)
        psf_base_norm = result_base['PSF'] / (result_base['PSF'].max() + 1e-20)
        ax.plot(X_UM, psf_base_norm, 'b--', lw=1.5, alpha=0.6, label='기준 (d=1, delta=0)')
        ax.plot(X_UM, psf_norm, 'r-', lw=2, label='현재 설계')
        # 자동 최적화 결과도 같이 표시
        if 'auto_opt' in st.session_state:
            ao = st.session_state['auto_opt']
            psf_opt = ao['result']['PSF']
            psf_opt_n = psf_opt / (psf_opt.max() + 1e-20)
            ax.plot(X_UM, psf_opt_n, 'g-', lw=2, alpha=0.8, label='자동 최적 설계')
        ax.set_xlim(-100, 100)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('정규화 강도')
        ax.set_title(f'PSF 비교  theta={theta}deg  |skew|={abs(result["skewness"]):.4f}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        thick_base = np.array(cp['thick_nm'])
        thick_cur = thick_base * d_scales
        x_pos = np.arange(n_layers)
        bw = 0.25
        ax.bar(x_pos - bw, thick_base, bw, label='기준', color='steelblue', alpha=0.7)
        ax.bar(x_pos, thick_cur, bw, label='현재', color='tomato', alpha=0.7)
        if 'auto_opt' in st.session_state:
            ao_ds = st.session_state['auto_opt']['d_scales']
            if len(ao_ds) == n_layers:
                thick_opt = thick_base * ao_ds
                ax.bar(x_pos + bw, thick_opt, bw, label='자동 최적', color='limegreen', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{cp["mats"][i]}\nL{i+1}' for i in range(n_layers)],
                           fontsize=8)
        ax.set_ylabel('두께 (nm)')
        ax.set_title('AR 코팅 층 두께 비교')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── delta_BM 자동 스윕 ──
    st.subheader('delta_BM 자동 스윕 (skewness 최적점 탐색)')
    delta_range = np.linspace(-30, 30, 61)
    skew_sweep = [full_pipeline(d_scales, d, theta, w1=w1, w2=w2,
                               coating_preset=coating_type)['skewness']
                  for d in delta_range]
    skew_sweep = np.array(skew_sweep)
    best_di = int(np.argmin(np.abs(skew_sweep)))
    best_dv = float(delta_range[best_di])
    best_sk = float(skew_sweep[best_di])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(delta_range, skew_sweep, 'b-', lw=2)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.axhline(0.10, color='orange', ls=':', lw=1.5, label='target +0.10')
        ax.axhline(-0.10, color='orange', ls=':', lw=1.5, label='target -0.10')
        ax.axvline(best_dv, color='r', ls='--', lw=2,
                   label=f'best delta={best_dv:.1f}um')
        ax.scatter([best_dv], [best_sk], c='red', s=100, zorder=5)
        ax.set_xlabel('delta_BM (um)')
        ax.set_ylabel('skewness')
        ax.set_title('현재 코팅 설정에서 delta_BM vs skewness')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_s2:
        st.metric('최적 delta_BM', f'{best_dv:.1f} μm')
        st.metric('해당 skewness', f'{best_sk:.4f}')
        pass_check = abs(best_sk) < 0.10
        if pass_check:
            st.success('목표 달성 (|skew| < 0.10)')
        else:
            st.warning(f'목표 미달 (|skew|={abs(best_sk):.3f} > 0.10)')

    # TMM 위상 곡선
    st.subheader('TMM 위상 분석')
    col_t1, col_t2 = st.columns(2)

    thetas_sweep = np.arange(0, 41, 1)
    T_arr = []
    dphi_arr = []
    for th_i in thetas_sweep:
        T_i, phi_i = tmm_phase_general(cp['thick_nm'], cp['ns'], d_scales, th_i)
        _, phi_0 = tmm_phase_general(cp['thick_nm'], cp['ns'], d_scales, 0.0)
        T_arr.append(T_i * 100)
        dphi_arr.append(np.degrees(phi_i - phi_0))

    with col_t1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(thetas_sweep, T_arr, 'b-o', lw=2, ms=3)
        ax.set_xlabel('입사각 theta (deg)')
        ax.set_ylabel('투과율 T (%)')
        ax.set_title('각도별 투과율 T(theta)')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_t2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(thetas_sweep, dphi_arr, 'r-o', lw=2, ms=3)
        ax.set_xlabel('입사각 theta (deg)')
        ax.set_ylabel('dphi (deg)')
        ax.set_title('각도별 위상 왜곡 dphi(theta)')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 탭 3: 데이터 흐름
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header('데이터 흐름도')
    st.markdown('> 기술보고서 2.4절 — 물리 엔진 → PINN → FNO → BoTorch')

    flow_md = """
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    설계 변수 입력                                │
    │   d1~d4 (AR 코팅)  +  Δx, Δy (BM 오프셋)  +  a1, a2, h, g      │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
    ┌────────────┐   Δφ(θ,λ)   ┌────────────┐   U_CG(x)   ┌────────────┐
    │   TMM      │────────────▶│   ASM      │────────────▶│   BM2      │
    │ AR 코팅     │  위상 계산   │ CG 550μm   │  파동 전파   │ 회절 격자   │
    │ 위상 엔진   │            │ n=1.52     │            │ w2=40μm    │
    └────────────┘            └────────────┘            └─────┬──────┘
                                                              │
                              ┌────────────┐                  │
                              │   ASM      │◀─────────────────┘
                              │ BM간격 30μm │  Interlayer 전파
                              └─────┬──────┘
                                    │
                                    ▼
                              ┌────────────┐
                              │   BM1      │  + delta_BM 오프셋
                              │ w1=30μm    │
                              └─────┬──────┘
                                    │
                                    ▼
                              ┌────────────┐
                              │ PSF 계산    │  I(x) = |U|² × T_coating
                              │ MTF, skew  │
                              └─────┬──────┘
                                    │
                          ┌─────────┼─────────┐
                          ▼         ▼         ▼
                      ┌──────┐ ┌──────┐ ┌──────────┐
                      │ MTF  │ │  T   │ │ skewness │
                      │  ↑   │ │  ↑   │ │   ↓      │
                      └──────┘ └──────┘ └──────────┘
                          │         │         │
                          ▼         ▼         ▼
                      ┌──────────────────────────┐
                      │   BoTorch qNEHVI         │
                      │   3목적 Pareto 최적화      │
                      └──────────────────────────┘
    ```
    """
    st.markdown(flow_md)

    st.subheader('모듈별 입출력 명세')
    st.table({
        '모듈': ['TMM', 'ASM (CG)', 'BM2 Mask', 'ASM (BM)', 'BM1 Mask', 'PSF 지표'],
        '입력': ['d_scales, θ, λ', 'U_in, Δφ', 'U_CG(x)', 'U_bm2(x)', 'U_bm(x), δ_BM',
                 'I(x)'],
        '출력': ['T, Δφ(θ)', 'U_CG(x)', 'U_CG × mask', 'U_bm(x)', 'U_out(x)',
                 'MTF, T, skew'],
        '계산 시간': ['~0.1ms', '~0.5ms', '~0.01ms', '~0.5ms', '~0.01ms', '~0.1ms'],
    })


# ══════════════════════════════════════════════════════════════════════════
# 탭 4: 핵심 수식
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header('핵심 수식')
    st.markdown('> 기술보고서 3.1절, 5.4절 — Helmholtz·PINN·FNO·qNEHVI')

    st.subheader('1. Helmholtz 방정식 (BM 근거리장)')
    st.latex(r'\nabla^2 E(x,z) + k_0^2 \, n^2(x,z) \, E(x,z) = 0')
    st.caption('k₀ = 2π/λ ≈ 7.39 μm⁻¹ (λ=850nm)')

    st.subheader('2. PINN 5항 손실 함수')
    st.latex(r'\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{PDE} + \lambda_2 \mathcal{L}_{BM} + \lambda_3 \mathcal{L}_{Fresnel} + \lambda_4 \mathcal{L}_{data} + \lambda_5 \mathcal{L}_{phase}^{\star}')

    st.markdown("""
    | Loss 항 | 수식 | 역할 |
    |---------|------|------|
    | L_PDE | ‖∇²U + k₀²n²U‖² | Helmholtz 방정식 잔차 |
    | L_BM | ‖U(x∈BM)‖² | BM 흡수 경계 |
    | L_Fresnel | ‖U_pred - U_Fresnel‖² | Fresnel 반사/투과 |
    | L_data | ‖U_pred - U_meas‖² | 관측 데이터 피팅 |
    | **L_phase★** | **‖∠U_pred - Δφ_TMM‖²** | **TMM 위상 주입 (핵심 혁신)** |
    """)

    st.subheader('3. SIREN 활성 함수')
    st.latex(r'\phi_i(x) = \sin(\omega_0 \cdot W_i \cdot x + b_i), \quad \omega_0 = 30')
    st.caption('sin 활성화 → 파동 현상의 고주파 성분을 자연스럽게 표현')

    st.subheader('4. FNO Spectral Convolution')
    st.latex(r'\hat{v}(k) = \text{FFT}[v(x)]')
    st.latex(r'\hat{y}(k) = R_\theta \cdot \hat{v}(k), \quad k \leq k_{max}')
    st.latex(r'\text{Output} = \sigma\left(\text{IFFT}[\hat{y}] + W \cdot v\right)')

    st.subheader('5. BoTorch qNEHVI')
    st.latex(r'\max_{X} \; \text{HVI}\left(\{\text{MTF}(X),\; T(X),\; -|\text{skewness}(X)|\}\right)')
    st.caption('3목적 동시 최적화: MTF↑, 광량↑, |skewness|↓')

    st.subheader('6. 이론적 최적 BM 오프셋')
    st.latex(r'\delta_{BM}^{*} \approx 0.169 \times d')
    st.caption('d = BM 전체 두께 (h₁+g+h₂). PINN 회절 해석으로 도출.')


# ══════════════════════════════════════════════════════════════════════════
# 탭 5: Pareto 탐색기
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header('Pareto 탐색기 + 자동 최적화 엔진')
    st.markdown('> Differential Evolution으로 5D 설계 공간 탐색 → Pareto Front 시각화')

    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        st.subheader('최적화 설정')
        theta_opt = st.selectbox('최적화 입사각', [15, 20, 25, 30], index=3)
        n_pop = st.slider('탐색 강도 (popsize)', 10, 30, 15, 5)
        n_gen = st.slider('세대 수 (maxiter)', 10, 50, 25, 5)
        w_mtf = st.slider('MTF 가중치', 0.0, 1.0, 0.3, 0.1)
        w_t = st.slider('T_total 가중치', 0.0, 1.0, 0.3, 0.1)
        w_skew = st.slider('skewness 가중치', 0.0, 1.0, 0.4, 0.1)

        if st.button('5D 최적 설계 탐색', type='primary', use_container_width=True):
            from scipy.optimize import differential_evolution

            progress = st.progress(0, '탐색 시작...')
            all_evals = []
            eval_count = [0]

            def multi_obj(params):
                ds = np.array(params[:4])
                db = params[4]
                r = full_pipeline(ds, db, theta_opt)
                all_evals.append({
                    'd_scales': ds.tolist(), 'delta_bm': float(db),
                    'MTF': r['MTF'], 'T': r['T_total'], 'skew': r['skewness']
                })
                eval_count[0] += 1
                if eval_count[0] % 20 == 0:
                    frac = min(eval_count[0] / (n_pop * 5 * n_gen), 1.0)
                    progress.progress(frac, f'평가 {eval_count[0]}개...')
                # 종합 점수 (모두 최대화 → 음수 반환)
                score = (w_mtf * r['MTF']
                         + w_t * r['T_total'] * 50
                         + w_skew * (1.0 - abs(r['skewness'])))
                return -score

            bounds_5d = [(0.5, 1.5)] * 4 + [(-30.0, 30.0)]
            opt = differential_evolution(
                multi_obj, bounds_5d, seed=42,
                maxiter=n_gen, popsize=n_pop,
                mutation=(0.5, 1.5), recombination=0.8, tol=1e-5
            )

            progress.empty()
            st.session_state['pareto_data'] = all_evals

            best_ds = np.array(opt.x[:4])
            best_db = opt.x[4]
            best_r = full_pipeline(best_ds, best_db, theta_opt)
            st.session_state['pareto_best'] = {
                'd_scales': best_ds, 'delta_bm': best_db, 'result': best_r
            }
            st.success(f'탐색 완료! 총 {len(all_evals)}개 평가')

    with col_p2:
        if 'pareto_data' in st.session_state:
            data = st.session_state['pareto_data']
            mtfs = np.array([d['MTF'] for d in data])
            ts = np.array([d['T'] for d in data])
            skews = np.array([abs(d['skew']) for d in data])

            # 파레토 판별
            is_pareto = np.ones(len(data), dtype=bool)
            obj = np.column_stack([mtfs, ts, -skews])
            for i in range(len(obj)):
                for j in range(len(obj)):
                    if i != j and np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                        is_pareto[i] = False
                        break

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            # MTF vs |skewness|
            ax = axes[0]
            ax.scatter(skews[~is_pareto], mtfs[~is_pareto],
                       c='lightgray', s=15, alpha=0.4, label='탐색 점')
            ax.scatter(skews[is_pareto], mtfs[is_pareto],
                       c='red', s=60, zorder=5, label='Pareto 최적')
            ax.axvline(0.10, color='orange', ls='--', lw=1.5, label='목표 0.10')
            ax.set_xlabel('|skewness|')
            ax.set_ylabel('MTF')
            ax.set_title('MTF vs |skewness|')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # MTF vs T
            ax = axes[1]
            ax.scatter(ts[~is_pareto], mtfs[~is_pareto],
                       c='lightgray', s=15, alpha=0.4)
            ax.scatter(ts[is_pareto], mtfs[is_pareto],
                       c='red', s=60, zorder=5)
            ax.set_xlabel('T_total')
            ax.set_ylabel('MTF')
            ax.set_title('MTF vs T_total')
            ax.grid(alpha=0.3)

            # T vs |skewness|
            ax = axes[2]
            ax.scatter(skews[~is_pareto], ts[~is_pareto],
                       c='lightgray', s=15, alpha=0.4)
            ax.scatter(skews[is_pareto], ts[is_pareto],
                       c='red', s=60, zorder=5)
            ax.set_xlabel('|skewness|')
            ax.set_ylabel('T_total')
            ax.set_title('T_total vs |skewness|')
            ax.grid(alpha=0.3)

            plt.suptitle(f'Pareto Front (theta={theta_opt}deg, {len(data)} 평가)',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── 최적 설계 상세 ──
            if 'pareto_best' in st.session_state:
                pb = st.session_state['pareto_best']
                br = pb['result']

                st.divider()
                st.subheader('최적 설계 결과')

                c1, c2, c3, c4 = st.columns(4)
                c1.metric('MTF', f'{br["MTF"]:.4f}')
                c2.metric('T_total', f'{br["T_total"]:.6f}')
                c3.metric('|skewness|', f'{abs(br["skewness"]):.4f}',
                          '목표 < 0.10' if abs(br['skewness']) < 0.10 else '목표 초과')
                c4.metric('delta_BM', f'{pb["delta_bm"]:.1f} um')

                # 최적 두께
                thick_opt = BASE_THICK_UM * pb['d_scales'] * 1e3
                thick_base = BASE_THICK_UM * 1e3

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown('**최적 AR 코팅 두께**')
                    for i, (mat, tb, to) in enumerate(zip(
                        ['SiO2', 'TiO2', 'SiO2', 'TiO2'], thick_base, thick_opt)):
                        change = (to - tb) / tb * 100
                        st.markdown(
                            f'Layer {i+1} ({mat}): '
                            f'{tb:.1f}nm -> **{to:.1f}nm** ({change:+.0f}%)')

                with col_r2:
                    # PSF 비교
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    r_base = full_pipeline(np.ones(4), 0.0, theta_opt)
                    psf_b = r_base['PSF'] / (r_base['PSF'].max() + 1e-20)
                    psf_o = br['PSF'] / (br['PSF'].max() + 1e-20)
                    ax.plot(X_UM, psf_b, 'b--', lw=1.5, label=f'기준 (|skew|={abs(r_base["skewness"]):.3f})')
                    ax.plot(X_UM, psf_o, 'r-', lw=2, label=f'최적 (|skew|={abs(br["skewness"]):.4f})')
                    ax.set_xlim(-80, 80)
                    ax.set_xlabel('x (um)')
                    ax.set_ylabel('정규화 강도')
                    ax.set_title('기준 vs 최적 PSF')
                    ax.legend(fontsize=9)
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # JSON 다운로드
                export = {
                    'theta_deg': theta_opt,
                    'optimal_d_scales': pb['d_scales'].tolist(),
                    'optimal_delta_bm': float(pb['delta_bm']),
                    'optimal_thickness_nm': thick_opt.tolist(),
                    'MTF': float(br['MTF']),
                    'T_total': float(br['T_total']),
                    'skewness': float(br['skewness']),
                    'total_evaluations': len(data),
                    'pareto_count': int(is_pareto.sum()),
                }
                st.download_button(
                    '최적 설계 JSON 다운로드',
                    json.dumps(export, ensure_ascii=False, indent=2),
                    'optimal_design_studio.json',
                    'application/json',
                    use_container_width=True
                )

            # 상위 10개 출력
            scores = mtfs * w_mtf + ts * 50 * w_t + (1 - skews) * w_skew
            top_n = min(10, len(data))
            top_idx = np.argsort(-scores)[:top_n]

            st.divider()
            st.subheader(f'상위 {top_n}개 설계')
            rows = []
            for rank, idx in enumerate(top_idx):
                d = data[idx]
                ds = d['d_scales']
                rows.append({
                    '순위': rank + 1,
                    'MTF': f'{d["MTF"]:.4f}',
                    'T_total': f'{d["T"]:.5f}',
                    'skewness': f'{d["skew"]:.4f}',
                    'delta_BM': f'{d["delta_bm"]:.1f}',
                    'd1': f'{ds[0]:.2f}', 'd2': f'{ds[1]:.2f}',
                    'd3': f'{ds[2]:.2f}', 'd4': f'{ds[3]:.2f}',
                })
            st.dataframe(rows, use_container_width=True)

        else:
            st.info('왼쪽에서 "5D 최적 설계 탐색" 버튼을 클릭하세요.')


# ══════════════════════════════════════════════════════════════════════════
# 탭 8: 지문 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════
with tab10:
    st.header('지문 이미지 시뮬레이션')
    st.markdown('> 실제 UDFPS 센서에서 보이는 지문 이미지를 시뮬레이션합니다')

    # 지문 이미지 로드
    fp_path = 'fingerprint_sample.png'
    if os.path.exists(fp_path):
        from PIL import Image

        FP_SIZE = 128
        FP_PIXEL_UM = 8.0
        FP_FOV = FP_SIZE * FP_PIXEL_UM

        img_raw = Image.open(fp_path).convert('L')
        img_rs = img_raw.resize((FP_SIZE, FP_SIZE), Image.LANCZOS)
        fp_2d = np.array(img_rs, dtype=np.float64) / 255.0
        if fp_2d.mean() > 0.5:
            fp_2d = 1.0 - fp_2d

        # 현재 사이드바 설계 변수 사용
        d_sc_fp = d_scales
        db_fp = delta_bm
        th_fp = theta

        # 입사각별 1D PSF 계산 (0~35도, 5도 간격)
        theta_fp_range = range(0, 36, 5)
        psf_fp_base = {}
        psf_fp_cur = {}
        for th_i in theta_fp_range:
            rb = full_pipeline(np.ones(n_layers), 0.0, float(th_i),
                               w1=w1, w2=w2, coating_preset=coating_type)
            rc = full_pipeline(d_sc_fp, db_fp, float(th_i),
                               w1=w1, w2=w2, coating_preset=coating_type)
            psf_fp_base[th_i] = rb['PSF']
            psf_fp_cur[th_i] = rc['PSF']

        # 센서 시뮬레이션 (행별로 해당 각도 PSF 적용)
        STACK_H = 600.0

        def sim_sensor(fp, psf_dict):
            ny, nx = fp.shape
            cy = ny // 2
            result = np.zeros_like(fp)
            for row in range(ny):
                dy = abs(row - cy) * FP_PIXEL_UM
                th_r = min(35, np.degrees(np.arctan(dy / STACK_H)))
                th_key = int(5 * round(th_r / 5))
                th_key = min(th_key, max(psf_dict.keys()))
                psf_1d = np.abs(psf_dict[th_key])
                c = len(psf_1d) // 2
                hw = min(30, c)
                k = psf_1d[c-hw:c+hw+1]
                if k.sum() > 0:
                    k = k / k.sum()
                result[row] = np.convolve(fp[row], k, mode='same')
            if result.max() > 0:
                result /= result.max()
            return result

        sensor_base_fp = sim_sensor(fp_2d, psf_fp_base)
        sensor_cur_fp = sim_sensor(fp_2d, psf_fp_cur)

        # 자동 최적 설계도 시뮬레이션
        has_auto_opt = 'auto_opt' in st.session_state
        if has_auto_opt:
            ao = st.session_state['auto_opt']
            ao_ds = np.array(ao['d_scales'])
            ao_db = ao['delta_bm']
            psf_fp_opt = {}
            for th_i in theta_fp_range:
                ro = full_pipeline(ao_ds, ao_db, float(th_i),
                                   w1=w1, w2=w2, coating_preset=coating_type)
                psf_fp_opt[th_i] = ro['PSF']
            sensor_opt_fp = sim_sensor(fp_2d, psf_fp_opt)

        # 시각화
        ext_fp = [-FP_FOV/2, FP_FOV/2, -FP_FOV/2, FP_FOV/2]
        n_cols = 5 if has_auto_opt else 4

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
        fig.suptitle(f'UDFPS 센서 지문 이미지  ({coating_type})',
                     fontsize=14, fontweight='bold')

        axes[0].imshow(fp_2d, cmap='gray', extent=ext_fp, origin='lower')
        axes[0].set_title('원본 지문\n(이상적)', fontsize=12, fontweight='bold')

        axes[1].imshow(sensor_base_fp, cmap='gray', extent=ext_fp, origin='lower')
        axes[1].set_title('기준 설계\n(d=1, delta=0)', fontsize=12, fontweight='bold', color='red')

        axes[2].imshow(sensor_cur_fp, cmap='gray', extent=ext_fp, origin='lower')
        axes[2].set_title('현재 설계\n(사이드바 값)', fontsize=12, fontweight='bold', color='blue')

        if has_auto_opt:
            axes[3].imshow(sensor_opt_fp, cmap='gray', extent=ext_fp, origin='lower')
            axes[3].set_title('자동 최적 설계\n(AI 탐색 결과)', fontsize=12, fontweight='bold', color='green')
            diff_fp = np.abs(sensor_opt_fp - sensor_base_fp)
            im_d = axes[4].imshow(diff_fp, cmap='hot', extent=ext_fp, origin='lower')
            axes[4].set_title('최적-기준\n차이 맵', fontsize=12, fontweight='bold')
            plt.colorbar(im_d, ax=axes[4], shrink=0.8)
        else:
            diff_fp = np.abs(sensor_cur_fp - sensor_base_fp)
            im_d = axes[3].imshow(diff_fp, cmap='hot', extent=ext_fp, origin='lower')
            axes[3].set_title('현재-기준\n차이 맵', fontsize=12, fontweight='bold')
            plt.colorbar(im_d, ax=axes[3], shrink=0.8)

        for ax in axes:
            ax.set_xlabel('x (um)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 정량 비교
        corr_b = float(np.corrcoef(fp_2d.ravel(), sensor_base_fp.ravel())[0, 1])
        corr_c = float(np.corrcoef(fp_2d.ravel(), sensor_cur_fp.ravel())[0, 1])

        if has_auto_opt:
            corr_o = float(np.corrcoef(fp_2d.ravel(), sensor_opt_fp.ravel())[0, 1])

            c1, c2, c3 = st.columns(3)
            c1.metric('기준 설계 (상관계수)', f'{corr_b:.4f}', '기준점')
            c2.metric('현재 설계 (슬라이더)', f'{corr_c:.4f}',
                      f'기준 대비 {corr_c - corr_b:+.4f}')
            c3.metric('자동 최적 설계 (AI)', f'{corr_o:.4f}',
                      f'기준 대비 {corr_o - corr_b:+.4f}')

            # 비교 테이블
            st.markdown(f"""
            | 설계 | 상관계수 | 기준 대비 | 비고 |
            |------|---------|----------|------|
            | 기준 (d=1, delta=0) | **{corr_b:.4f}** | - | baseline |
            | 현재 (슬라이더) | **{corr_c:.4f}** | {corr_c-corr_b:+.4f} | 수동 조절 |
            | **자동 최적 (AI)** | **{corr_o:.4f}** | **{corr_o-corr_b:+.4f}** | delta={ao_db:.1f}um |
            """)

            st.caption(f'자동 최적 설계: delta_BM={ao_db:.1f}um, '
                       f'd_scales={[round(v,2) for v in ao_ds.tolist()]}')
        else:
            c1, c2 = st.columns(2)
            c1.metric('기준 설계 (상관계수)', f'{corr_b:.4f}', '기준점')
            c2.metric('현재 설계 (슬라이더)', f'{corr_c:.4f}',
                      f'기준 대비 {corr_c - corr_b:+.4f}')
            st.info('탭 2에서 "최적 설계 자동 탐색"을 실행하면 최적 설계 지문도 비교됩니다.')
    else:
        st.warning('fingerprint_sample.png 파일이 없습니다. '
                   '프로젝트 폴더에 지문 이미지를 추가하세요.')


# ══════════════════════════════════════════════════════════════════════════
# 탭 9: 로드맵
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.header('개발 로드맵')
    st.markdown('> 기술보고서 6.1절 — Phase별 개발 전략')

    roadmap = {
        'Phase': ['Phase 1 PoC', 'Phase 2 Physics', 'Phase 3 Surrogate', 'Phase 4 Agentic'],
        '기간': ['즉시~3M', '3~6M', '6~9M', '9~12M'],
        'UDFPS 핵심 작업': [
            'TMM Δφ 자동화\nLightTools 수집\nASM CG 전파 구현',
            'Helmholtz PINN\nSIREN+FFE 적용\nL_phase Loss 통합',
            'FNO Surrogate\nPINN 증류 <1ms\nBoTorch qNEHVI',
            'Design Studio UI\nFastAPI+WebSocket\nCelery Job Queue'
        ],
        'KPI': [
            'TMM-Zemax 오차<5%\nMTF 예측 오차<10%',
            '비대칭 PSF 재현\nδ_BM* 이론값 검증',
            'FNO 추론<1ms\nPareto Front 수렴',
            '설계 사이클 90%↓\n사람 개입 최소화'
        ],
        '상태': ['✅ 완료', '✅ 완료', '✅ 완료', '🔄 진행중']
    }
    st.table(roadmap)

    st.subheader('Phase 1 달성 성과')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('PSF skewness', '0.075', '-79% (초기 0.354)')
    c2.metric('MTF@ridge', '99.78%', '목표 >60% 초과달성')
    c3.metric('설계 사이클', '수분', '기존 수일~수주')
    c4.metric('최종 목표', '<0.10 달성', 'Phase 1에서 조기달성')

    st.subheader('투자 대비 효과 (ROI)')
    roi = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        '주요 도입': ['MLP Surrogate', 'PINN+Bayesian', 'FNO+BoTorch', 'Agentic Loop'],
        '시간 절감': ['월 800시간', '월 1,200시간', '월 1,700시간', '월 2,000시간+'],
        '절감률': ['36%', '55%', '77%', '90%+'],
    }
    st.table(roi)

    st.info('Phase 1 Break-even: 3개월 | 1년 ROI: 10×')


# ══════════════════════════════════════════════════════════════════════════
# 탭 3: 다각도 Robustness
# ══════════════════════════════════════════════════════════════════════════
with tab7:
    st.header('다각도 Robustness 평가')
    st.markdown('> 0~40° 전체 각도에서 현재 설계의 성능 프로파일')

    # 현재 사이드바 설계 변수 사용
    thetas_rob = np.arange(0, 41, 2)
    mtf_rob, t_rob, skew_rob, dphi_rob = [], [], [], []

    for th_r in thetas_rob:
        r = full_pipeline(d_scales, delta_bm, th_r, w1=w1, w2=w2,
                          coating_preset=coating_type)
        mtf_rob.append(r['MTF'])
        t_rob.append(r['T_total'])
        skew_rob.append(r['skewness'])
        dphi_rob.append(r['dphi_deg'])

    # DX vs DX+ 비교 (다른 코팅으로 같은 설계)
    other_coating = [k for k in COATING_PRESETS if k != coating_type][0]
    other_cp = COATING_PRESETS[other_coating]
    d_other = np.ones(len(other_cp['thick_nm']))
    skew_other = []
    for th_r in thetas_rob:
        r2 = full_pipeline(d_other, delta_bm, th_r, w1=w1, w2=w2,
                           coating_preset=other_coating)
        skew_other.append(r2['skewness'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'다각도 Robustness  ({coating_type}, delta_BM={delta_bm:.0f}um)',
                 fontsize=14, fontweight='bold')

    # MTF vs theta
    ax = axes[0, 0]
    ax.plot(thetas_rob, mtf_rob, 'b-o', lw=2, ms=5)
    ax.axhline(0.6, color='orange', ls='--', lw=1.5, label='목표 >0.60')
    ax.fill_between(thetas_rob, 0, mtf_rob, alpha=0.15, color='blue')
    ax.set_xlabel('입사각 (deg)'); ax.set_ylabel('MTF')
    ax.set_title('MTF vs 입사각'); ax.legend(); ax.grid(alpha=0.3)

    # T vs theta
    ax = axes[0, 1]
    ax.plot(thetas_rob, t_rob, 'g-s', lw=2, ms=5)
    ax.fill_between(thetas_rob, 0, t_rob, alpha=0.15, color='green')
    ax.set_xlabel('입사각 (deg)'); ax.set_ylabel('T_total')
    ax.set_title('T_total vs 입사각'); ax.grid(alpha=0.3)

    # skewness vs theta (DX vs DX+ 비교)
    ax = axes[1, 0]
    ax.plot(thetas_rob, skew_rob, 'r-o', lw=2, ms=5, label=coating_type)
    ax.plot(thetas_rob, skew_other, 'b--s', lw=1.5, ms=4, alpha=0.6,
            label=f'{other_coating} (기준)')
    ax.axhline(0.10, color='orange', ls=':', lw=1.5)
    ax.axhline(-0.10, color='orange', ls=':', lw=1.5)
    ax.axhspan(-0.10, 0.10, alpha=0.08, color='green', label='목표 범위')
    ax.set_xlabel('입사각 (deg)'); ax.set_ylabel('skewness')
    ax.set_title('skewness vs 입사각 (코팅 비교)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 위상 왜곡 vs theta
    ax = axes[1, 1]
    ax.plot(thetas_rob, dphi_rob, 'm-^', lw=2, ms=5)
    ax.set_xlabel('입사각 (deg)'); ax.set_ylabel('dphi (deg)')
    ax.set_title('위상 왜곡 dphi vs 입사각'); ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Robustness 점수
    skew_arr_rob = np.abs(np.array(skew_rob))
    worst_skew = float(skew_arr_rob.max())
    worst_angle = int(thetas_rob[np.argmax(skew_arr_rob)])
    avg_skew = float(skew_arr_rob.mean())
    pass_angles = int((skew_arr_rob < 0.10).sum())

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Worst |skew|', f'{worst_skew:.4f}', f'theta={worst_angle}deg')
    c2.metric('평균 |skew|', f'{avg_skew:.4f}')
    c3.metric('목표 통과 각도', f'{pass_angles}/{len(thetas_rob)}')
    c4.metric('Robustness 점수', f'{pass_angles/len(thetas_rob)*100:.0f}%')

    if worst_skew < 0.10:
        st.success('모든 각도에서 |skewness| < 0.10 목표 달성!')
    else:
        st.warning(f'theta={worst_angle}deg 에서 |skew|={worst_skew:.3f} 초과. '
                   f'delta_BM 조절 필요.')


# ══════════════════════════════════════════════════════════════════════════
# 탭 4: 민감도 분석
# ══════════════════════════════════════════════════════════════════════════
with tab8:
    st.header('민감도 분석')
    st.markdown('> 각 설계 변수가 성능에 미치는 영향력 정량화')

    st.subheader('One-at-a-time (OAT) 민감도')

    # 기준점
    base_r = full_pipeline(d_scales, delta_bm, theta, w1=w1, w2=w2,
                           coating_preset=coating_type)
    perturbation = 0.1  # 10% 변동

    var_names = [f'd{i+1} ({cp["mats"][i]})' for i in range(n_layers)]
    var_names += ['delta_BM', 'w1', 'w2']

    sens_mtf, sens_skew, sens_t = [], [], []

    for vi in range(n_layers + 3):
        ds_p = d_scales.copy()
        db_p, w1_p, w2_p = delta_bm, w1, w2

        if vi < n_layers:
            ds_p[vi] *= (1 + perturbation)
        elif vi == n_layers:
            db_p += 3.0  # delta_BM +3um
        elif vi == n_layers + 1:
            w1_p *= (1 + perturbation)
        else:
            w2_p *= (1 + perturbation)

        r_p = full_pipeline(ds_p, db_p, theta, w1=w1_p, w2=w2_p,
                            coating_preset=coating_type)

        sens_mtf.append(abs(r_p['MTF'] - base_r['MTF']))
        sens_skew.append(abs(r_p['skewness'] - base_r['skewness']))
        sens_t.append(abs(r_p['T_total'] - base_r['T_total']))

    # 정규화
    sens_mtf = np.array(sens_mtf)
    sens_skew = np.array(sens_skew)
    sens_t = np.array(sens_t)
    if sens_mtf.max() > 0: sens_mtf_n = sens_mtf / sens_mtf.max()
    else: sens_mtf_n = sens_mtf
    if sens_skew.max() > 0: sens_skew_n = sens_skew / sens_skew.max()
    else: sens_skew_n = sens_skew
    if sens_t.max() > 0: sens_t_n = sens_t / sens_t.max()
    else: sens_t_n = sens_t

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    y_pos = np.arange(len(var_names))
    colors = ['#2196F3'] * n_layers + ['#F44336', '#FF9800', '#FF9800']

    ax = axes[0]
    bars = ax.barh(y_pos, sens_mtf_n, color=colors, alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(var_names)
    ax.set_xlabel('정규화 민감도'); ax.set_title('MTF 민감도')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1]
    bars = ax.barh(y_pos, sens_skew_n, color=colors, alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(var_names)
    ax.set_xlabel('정규화 민감도'); ax.set_title('Skewness 민감도')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[2]
    bars = ax.barh(y_pos, sens_t_n, color=colors, alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(var_names)
    ax.set_xlabel('정규화 민감도'); ax.set_title('T_total 민감도')
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'민감도 분석 (theta={theta}deg, 10% 변동 기준)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 가장 영향 큰 변수
    top_skew_var = var_names[np.argmax(sens_skew)]
    top_mtf_var = var_names[np.argmax(sens_mtf)]

    st.divider()
    col_s1, col_s2 = st.columns(2)
    col_s1.info(f'**Skewness에 가장 영향 큰 변수:** {top_skew_var}')
    col_s2.info(f'**MTF에 가장 영향 큰 변수:** {top_mtf_var}')

    # 2D 스윕 히트맵
    st.subheader('2D 민감도 히트맵 (delta_BM vs 입사각)')
    thetas_2d = np.arange(0, 41, 5)
    deltas_2d = np.linspace(-25, 25, 21)
    skew_2d = np.zeros((len(thetas_2d), len(deltas_2d)))

    for i, th2 in enumerate(thetas_2d):
        for j, db2 in enumerate(deltas_2d):
            r2d = full_pipeline(d_scales, db2, th2, w1=w1, w2=w2,
                                coating_preset=coating_type)
            skew_2d[i, j] = r2d['skewness']

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(skew_2d, extent=[deltas_2d[0], deltas_2d[-1],
                   thetas_2d[-1], thetas_2d[0]],
                   aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label='skewness')
    # 목표 영역 표시
    ax.contour(deltas_2d, thetas_2d, skew_2d,
               levels=[-0.10, 0.10], colors='lime', linewidths=2)
    ax.set_xlabel('delta_BM (um)')
    ax.set_ylabel('입사각 (deg)')
    ax.set_title(f'skewness 히트맵 ({coating_type})  초록선 = 목표 범위')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption('초록색 등고선 안쪽이 |skewness| < 0.10 목표 달성 영역')


# ══════════════════════════════════════════════════════════════════════════
# 탭 8: 설계 이력
# ══════════════════════════════════════════════════════════════════════════
with tab9:
    st.header('설계 이력 관리')
    st.markdown('> 저장된 설계를 비교하고 최적 설계를 선택합니다')

    history = st.session_state.get('design_history', [])

    if len(history) == 0:
        st.info('아직 저장된 설계가 없습니다. 탭 2에서 설계를 조절한 후 '
                '"현재 설계 저장" 버튼을 클릭하세요.')
    else:
        st.subheader(f'저장된 설계: {len(history)}개')

        # 테이블
        rows = []
        for i, h in enumerate(history):
            rows.append({
                '#': i + 1,
                '시간': h.get('time', ''),
                '코팅': h.get('coating', 'DX'),
                'MTF': h.get('MTF', 0),
                'T_total': h.get('T_total', 0),
                'skewness': h.get('skewness', 0),
                'delta_BM': h.get('delta_bm', 0),
                'theta': h.get('theta', 30),
            })
        st.dataframe(rows, use_container_width=True)

        # 비교 시각화
        if len(history) >= 2:
            st.subheader('설계 비교')
            mtfs_h = [h.get('MTF', 0) for h in history]
            skews_h = [abs(h.get('skewness', 0)) for h in history]
            ts_h = [h.get('T_total', 0) for h in history]
            labels_h = [f'#{i+1}' for i in range(len(history))]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            ax = axes[0]
            colors_h = ['limegreen' if m > 0.99 else 'steelblue' for m in mtfs_h]
            ax.bar(labels_h, mtfs_h, color=colors_h, alpha=0.8)
            ax.set_ylabel('MTF'); ax.set_title('MTF 비교')
            ax.grid(axis='y', alpha=0.3)

            ax = axes[1]
            colors_h = ['limegreen' if s < 0.10 else 'tomato' for s in skews_h]
            ax.bar(labels_h, skews_h, color=colors_h, alpha=0.8)
            ax.axhline(0.10, color='orange', ls='--', lw=1.5, label='목표 0.10')
            ax.set_ylabel('|skewness|'); ax.set_title('|skewness| 비교')
            ax.legend(); ax.grid(axis='y', alpha=0.3)

            ax = axes[2]
            ax.bar(labels_h, ts_h, color='steelblue', alpha=0.8)
            ax.set_ylabel('T_total'); ax.set_title('T_total 비교')
            ax.grid(axis='y', alpha=0.3)

            plt.suptitle('설계 이력 비교', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 최고 설계 표시
            best_idx = int(np.argmin(skews_h))
            bh = history[best_idx]
            st.success(
                f'최고 설계: #{best_idx+1} — '
                f'MTF={bh.get("MTF",0):.4f}, '
                f'|skew|={abs(bh.get("skewness",0)):.4f}, '
                f'delta_BM={bh.get("delta_bm",0):.1f}um'
            )

        # 이력 초기화
        if st.button('이력 전체 삭제', type='secondary'):
            st.session_state['design_history'] = []
            st.rerun()


# ── 하단 ──────────────────────────────────────────────────────────────────
st.divider()
st.caption('UDFPS COE Design Studio v0.2 | Physics-Informed AI 설계 플랫폼 | 대외비 3등급')
