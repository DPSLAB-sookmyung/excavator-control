"""
KOCETI 30ton Excavator - PID + Feedforward Control Simulation
=============================================================
실험 흐름:
  1. 기구학 (FK, IK)
  2. Method1: IK 정밀도 오차       → 이론적 한계 (feedforward로 줄일 수 있음)
  3. Method2: 현재 경로 오차       → 제어 없는 실제 상태
  4. Method4: PID + Feedforward   → 피드백 + 선제 보상

굴착기 동역학 시뮬 모델:
  u [-1,1] → 각속도 = u * MAX_VEL → 관절각 적분
  (실제 유압 지연/dead-zone은 단순화)

주요 파라미터:
  - MAX_VEL: 실측 데이터 기반 각속도 상한
  - Kp, Ki, Kd: PID 게인
  - Kff: Feedforward 게인 (목표 각속도 기반)

게인 최적화 결과 (meta_1046.csv 기준, 그리드 탐색):
  - 1,440가지 조합 시뮬레이션으로 RMSE 최소값 탐색
  - 시뮬 기준 최적값이며 실제 기계에서는 재조정 필요
  - 실제 적용 순서: Kp → Kff → Ki → Kd 순으로 조금씩 조정

주의:
  - 아래 게인은 단순화된 시뮬 모델 기준임
  - MAX_VEL이 실측(Boom 6.4, Arm 20.5)보다 높게 설정됨
  - 실제 기계에서는 유압 응답 특성에 맞게 재튜닝 필요
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

DATA_PATH  = 'data/meta_1046.csv'
OUTPUT_DIR = '.'

a2 = 6245; a3 = 3113; a4 = 1910

DT = 0.1

# 굴착기 동역학 모델 파라미터
# 그리드 탐색으로 찾은 시뮬 최적값 (실측: Boom 6.4, Arm 20.5 deg/s)
MAX_VEL_BOOM = 12.0   # deg/s
MAX_VEL_ARM  = 35.0   # deg/s

# PID + Feedforward 게인
# 그리드 탐색 결과: 1,440가지 조합 중 RMSE 최소값
# 실제 기계 적용 시 Kp → Kff → Ki → Kd 순으로 재조정 필요
Kp       = 0.3
Ki       = 0.01
Kd       = 0.01
Kff_boom = 0.08   # Boom: 목표 각속도 기반 FF 게인
Kff_arm  = 0.02   # Arm:  목표 각속도 기반 FF 게인

# ── 1. 기구학 함수 ───────────────────────────────────────────────────────────

def fk(boom_deg, arm_deg, bkt_deg):
    """순기구학 (FK): 관절각 → 버킷 끝점 위치 (시상면)"""
    b = np.deg2rad(boom_deg); a = np.deg2rad(arm_deg); k = np.deg2rad(bkt_deg)
    reach = a2*np.cos(b) + a3*np.cos(b+a) + a4*np.cos(b+a+k)
    z     = a2*np.sin(b) + a3*np.sin(b+a) + a4*np.sin(b+a+k)
    return reach, z

def ik(reach, z, boom_deg, arm_deg, bkt_deg):
    """역기구학 (IK): 버킷 끝점 위치 → 관절각 (도달 불가 시 None, None)"""
    bkt_abs = np.deg2rad(boom_deg + arm_deg + bkt_deg)
    xw = reach - a4*np.cos(bkt_abs); zw = z - a4*np.sin(bkt_abs)
    D  = (xw**2 + zw**2 - a2**2 - a3**2) / (2*a2*a3)
    if abs(D) > 1.0: return None, None
    arm  = np.degrees(np.arctan2(-np.sqrt(max(0, 1-D**2)), D))
    boom = np.degrees(np.arctan2(zw, xw) -
                      np.arctan2(a3*np.sin(np.deg2rad(arm)),
                                 a2+a3*np.cos(np.deg2rad(arm))))
    return boom, arm

# ── 2. 제어기 클래스 ─────────────────────────────────────────────────────────

class FeedforwardPIDController:
    """
    PID + Feedforward 제어기
    u_ff    = Kff * dθ_ref/dt        ← 목표 각속도 기반 선제 보상
    u_pid   = Kp*e + Ki*∫e + Kd*de  ← 오차 피드백
    u_total = u_ff + u_pid

    FF 기여도 (meta_1046 기준):
      평상시   Boom 21.6%, Arm 29.5%
      고속구간 Boom 44.1%, Arm 79.4%  ← 방향전환/굴착시작 구간
    """
    def __init__(self, kp, ki, kd, kff, dt,
                 integral_limit=5.0, u_min=-1.0, u_max=1.0):
        self.Kp=kp; self.Ki=ki; self.Kd=kd; self.Kff=kff; self.dt=dt
        self.integral_limit=integral_limit
        self.u_min=u_min; self.u_max=u_max
        self.integral=0.0; self.prev_error=0.0; self.prev_ref=None

    def reset(self):
        self.integral=0.0; self.prev_error=0.0; self.prev_ref=None

    def update(self, theta_ref, theta_actual):
        error = theta_ref - theta_actual
        if self.prev_ref is None: self.prev_ref = theta_ref
        u_ff = self.Kff * (theta_ref - self.prev_ref) / self.dt
        self.prev_ref = theta_ref
        self.integral = np.clip(self.integral + error*self.dt,
                                -self.integral_limit, self.integral_limit)
        deriv = (error - self.prev_error) / self.dt
        self.prev_error = error
        u_pid = self.Kp*error + self.Ki*self.integral + self.Kd*deriv
        return np.clip(u_ff + u_pid, self.u_min, self.u_max)

# ── 3. 굴착기 동역학 시뮬레이션 ──────────────────────────────────────────────

def simulate(df, ctrl_boom, ctrl_arm,
             max_vel_boom=MAX_VEL_BOOM, max_vel_arm=MAX_VEL_ARM):
    ctrl_boom.reset(); ctrl_arm.reset()
    sim_boom = df['slope_boom'].iloc[0]
    sim_arm  = df['slope_arm'].iloc[0]
    booms, arms = [], []
    for _, row in df.iterrows():
        boom_ref = row['ik_boom'] if row['ik_boom'] is not None else sim_boom
        arm_ref  = row['ik_arm']  if row['ik_arm']  is not None else sim_arm
        u_boom = ctrl_boom.update(boom_ref, sim_boom)
        u_arm  = ctrl_arm.update(arm_ref,  sim_arm)
        sim_boom += np.clip(u_boom*max_vel_boom, -max_vel_boom, max_vel_boom)*DT
        sim_arm  += np.clip(u_arm *max_vel_arm,  -max_vel_arm,  max_vel_arm )*DT
        booms.append(sim_boom); arms.append(sim_arm)
    return np.array(booms), np.array(arms)

def path_rmse(boom_arr, arm_arr, bkt_series, ref_reach, ref_z):
    errs = []
    for b, a, k, rr, rz in zip(boom_arr, arm_arr, bkt_series, ref_reach, ref_z):
        r_, z_ = fk(b, a, k)
        errs.append(np.sqrt((r_-rr)**2 + (z_-rz)**2))
    return np.sqrt(np.mean(np.array(errs)**2)), np.array(errs)

# ── 4. 데이터 로드 ───────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH, skipinitialspace=True)
N  = len(df); t = np.arange(N)*DT

df['ref_reach'] = np.sqrt(df['bucket_x']**2 + df['bucket_y']**2)
df['ref_z']     = df['bucket_z']

print(f"  Samples : {N},  Duration: {t[-1]:.0f}s")
print(f"  Boom    : {df['slope_boom'].min():.1f} ~ {df['slope_boom'].max():.1f} deg")
print(f"  Arm     : {df['slope_arm'].min():.1f} ~ {df['slope_arm'].max():.1f} deg")

# ── 5. IK 기준 관절각 ────────────────────────────────────────────────────────

print("\n[IK] Computing reference joint angles...")
ik_res = df.apply(lambda r: ik(r['ref_reach'], r['ref_z'],
    r['slope_boom'], r['slope_arm'], r['slope_bucket']), axis=1)
df['ik_boom'] = [r[0] for r in ik_res]
df['ik_arm']  = [r[1] for r in ik_res]
print(f"  IK success: {df['ik_boom'].notna().sum()}/{N}")

# ── 6. Method1: IK 정밀도 오차 ───────────────────────────────────────────────

print("\n[Method1] IK precision error...")
m1_reach, m1_z = zip(*df.apply(
    lambda r: fk(r['ik_boom'], r['ik_arm'], r['slope_bucket'])
    if r['ik_boom'] is not None else (np.nan, np.nan), axis=1))
df['m1_reach']=m1_reach; df['m1_z']=m1_z
df['err_m1'] = np.sqrt((df['m1_reach']-df['ref_reach'])**2 +
                        (df['m1_z']   -df['ref_z']    )**2)
rmse_m1 = np.sqrt((df['err_m1']**2).mean())
print(f"  RMSE: {rmse_m1:.1f} mm  (이론적 한계)")

# ── 7. Method2: 현재 경로 오차 ───────────────────────────────────────────────

print("\n[Method2] Actual path error...")
m2_reach, m2_z = zip(*df.apply(
    lambda r: fk(r['slope_boom'], r['slope_arm'], r['slope_bucket']), axis=1))
df['m2_reach']=m2_reach; df['m2_z']=m2_z
df['err_m2'] = np.sqrt((df['m2_reach']-df['ref_reach'])**2 +
                        (df['m2_z']   -df['ref_z']    )**2)
rmse_m2 = np.sqrt((df['err_m2']**2).mean())
print(f"  RMSE: {rmse_m2:.1f} mm  (제어 없는 현재 상태)")

# ── 8. Method4: PID + Feedforward 시뮬 ───────────────────────────────────────

print("\n[Method4] PID + Feedforward simulation...")
ff_boom = FeedforwardPIDController(kp=Kp, ki=Ki, kd=Kd, kff=Kff_boom, dt=DT)
ff_arm  = FeedforwardPIDController(kp=Kp, ki=Ki, kd=Kd, kff=Kff_arm,  dt=DT)
m4_booms, m4_arms = simulate(df, ff_boom, ff_arm)
df['m4_boom']=m4_booms; df['m4_arm']=m4_arms
rmse_m4, err_m4_arr = path_rmse(m4_booms, m4_arms,
                                 df['slope_bucket'], df['ref_reach'], df['ref_z'])
df['err_m4']   = err_m4_arr
df['m4_reach'] = [fk(b,a,k)[0] for b,a,k in zip(m4_booms, m4_arms, df['slope_bucket'])]
df['m4_z']     = [fk(b,a,k)[1] for b,a,k in zip(m4_booms, m4_arms, df['slope_bucket'])]
print(f"  RMSE: {rmse_m4:.1f} mm  (Kp={Kp}, Ki={Ki}, Kd={Kd}, "
      f"Kff_boom={Kff_boom}, Kff_arm={Kff_arm})")

# ── 9. FF 기여도 분석 ────────────────────────────────────────────────────────

df['ik_boom_vel'] = df['ik_boom'].diff() / DT
df['ik_arm_vel']  = df['ik_arm'].diff()  / DT
df['err_boom']    = df['ik_boom'] - df['slope_boom']
df['err_arm']     = df['ik_arm']  - df['slope_arm']

u_pid_boom = Kp * df['err_boom'].abs().mean()
u_pid_arm  = Kp * df['err_arm'].abs().mean()
u_ff_boom  = Kff_boom * df['ik_boom_vel'].abs().mean()
u_ff_arm   = Kff_arm  * df['ik_arm_vel'].abs().mean()

fast_boom  = df[df['ik_boom_vel'].abs() > df['ik_boom_vel'].abs().quantile(0.9)]
fast_arm   = df[df['ik_arm_vel'].abs()  > df['ik_arm_vel'].abs().quantile(0.9)]
ff_fast_b  = (Kff_boom * fast_boom['ik_boom_vel'].abs()).mean()
pid_fast_b = (Kp * fast_boom['err_boom'].abs()).mean()
ff_fast_a  = (Kff_arm  * fast_arm['ik_arm_vel'].abs()).mean()
pid_fast_a = (Kp * fast_arm['err_arm'].abs()).mean()

# ── 10. 결과 요약 ─────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("  RESULTS SUMMARY")
print("="*62)
print(f"  Method1: IK precision (theoretical limit)  : {rmse_m1:.1f} mm")
print(f"  Method2: Actual path error (no control)    : {rmse_m2:.1f} mm")
print(f"  Method4: PID + Feedforward simulation      : {rmse_m4:.1f} mm")
print(f"  {'─'*54}")
print(f"  Improvement  (M2 → M4) : {rmse_m2-rmse_m4:.1f} mm")
print(f"  Remaining gap (M4 → M1): {rmse_m4-rmse_m1:.1f} mm")
print(f"\n  [FF 기여도 분석]")
print(f"  평상시   Boom FF/PID: {u_ff_boom/u_pid_boom*100:.1f}%,  "
      f"Arm FF/PID: {u_ff_arm/u_pid_arm*100:.1f}%")
print(f"  고속구간 Boom FF/PID: {ff_fast_b/pid_fast_b*100:.1f}%, "
      f"Arm FF/PID: {ff_fast_a/pid_fast_a*100:.1f}%")
print(f"\n  [게인 튜닝 가이드 - 실제 기계 적용 시]")
print(f"  1. Kp부터 조금씩 올려서 진동 없이 추종하는 값 찾기")
print(f"  2. Kff를 올려서 빠른 구간 오차 줄이기")
print(f"  3. Ki로 정상상태 오차 제거")
print(f"  4. Kd로 진동 억제")
print(f"\n  [주의] 아래 게인은 시뮬 기준 최적값")
print(f"  실제 기계에서는 유압 응답에 맞게 재조정 필요")
print("="*62)

# ── 11. 그래프 ────────────────────────────────────────────────────────────────

print("\n[Plotting...]")
matplotlib.rcParams['axes.unicode_minus'] = False

C_TARGET = '#444441'
C_M1     = '#0F6E56'
C_M2     = '#993C1D'
C_M4     = '#534AB7'

fig = plt.figure(figsize=(15, 14))
fig.suptitle('KOCETI 30ton — PID + Feedforward Control Simulation',
             fontsize=13, fontweight='bold')
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# ── 그래프1. 시상면 궤적 비교 ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
ax.plot(df['ref_reach']/1000, df['ref_z']/1000,
        color=C_TARGET, lw=2.2, label='Target path', zorder=6)
ax.plot(df['m2_reach']/1000, df['m2_z']/1000,
        color=C_M2, lw=0.9, ls=':', alpha=0.7,
        label=f'No control  RMSE={rmse_m2:.0f} mm')
ax.plot(df['m4_reach']/1000, df['m4_z']/1000,
        color=C_M4, lw=1.2, ls='-.', alpha=0.85,
        label=f'PID + FF  RMSE={rmse_m4:.0f} mm')
ax.plot(df['m1_reach']/1000, df['m1_z']/1000,
        color=C_M1, lw=1.0, ls='--', alpha=0.7,
        label=f'IK limit  RMSE={rmse_m1:.0f} mm')
ax.set_xlabel('Reach [m]'); ax.set_ylabel('Z [m]')
ax.set_title('1. Sagittal Plane Trajectory Comparison')
ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

# ── 그래프2. 경로 오차 시계열 ────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
ax.plot(t, df['err_m2'], color=C_M2, lw=0.9, alpha=0.7,
        label=f'No control  RMSE={rmse_m2:.0f} mm')
ax.plot(t, df['err_m4'], color=C_M4, lw=0.9, alpha=0.9,
        label=f'PID + FF  RMSE={rmse_m4:.0f} mm')
ax.axhline(rmse_m1, color=C_M1, ls='--', lw=1.5, alpha=0.8,
           label=f'IK limit  {rmse_m1:.0f} mm')
ax.axhline(rmse_m2, color=C_M2, ls='--', lw=1.0, alpha=0.5)
ax.axhline(rmse_m4, color=C_M4, ls='--', lw=1.0, alpha=0.5)
ax.fill_between(t, df['err_m4'], df['err_m2'],
                alpha=0.10, color='orange',
                label=f'Improvement: {rmse_m2-rmse_m4:.0f} mm')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Path error [mm]')
ax.set_title('2. Path Following Error over Time')
ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

# ── 그래프3. Boom 관절각: IMU vs IK 기준 vs PID+FF 시뮬 ─────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, df['slope_boom'], color=C_TARGET, lw=1.2, label='IMU actual', zorder=5)
ax.plot(t, df['ik_boom'],    color=C_M1,     lw=1.0, ls='--',
        label='IK reference', alpha=0.8)
ax.plot(t, df['m4_boom'],    color=C_M4,     lw=1.0, ls=':',
        label='PID+FF sim', alpha=0.85)
ax.fill_between(t, df['slope_boom'], df['ik_boom'],
                alpha=0.12, color=C_M1, label='Current error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title('3. Boom: IMU / IK reference / PID+FF sim')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

# ── 그래프4. Arm 관절각: IMU vs IK 기준 vs PID+FF 시뮬 ──────────────────────
ax = fig.add_subplot(gs[2, 1])
ax.plot(t, df['slope_arm'], color=C_TARGET, lw=1.2, label='IMU actual', zorder=5)
ax.plot(t, df['ik_arm'],    color=C_M1,     lw=1.0, ls='--',
        label='IK reference', alpha=0.8)
ax.plot(t, df['m4_arm'],    color=C_M4,     lw=1.0, ls=':',
        label='PID+FF sim', alpha=0.85)
ax.fill_between(t, df['slope_arm'], df['ik_arm'],
                alpha=0.12, color=C_M1, label='Current error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title('4. Arm: IMU / IK reference / PID+FF sim')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

out_path = f'{OUTPUT_DIR}/excavator_pid_ff.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

# ── 12. 확대본 (갭이 가장 큰 구간) ───────────────────────────────────────────
df['gap_2d'] = np.sqrt((df['ik_boom']-df['slope_boom'])**2 +
                        (df['ik_arm'] -df['slope_arm'] )**2)
peak_idx = df['gap_2d'].rolling(50, center=True).mean().idxmax()
s = max(0,   peak_idx - 100)
e = min(N-1, peak_idx + 100)
t_s, t_e = s*DT, e*DT
ts = t[s:e]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(f'Zoom-in: Largest Error Region  ({t_s:.0f}s ~ {t_e:.0f}s)',
              fontsize=12, fontweight='bold')

ax = axes2[0]
ax.plot(ts, df['slope_boom'].iloc[s:e], color=C_TARGET, lw=1.5,
        label='IMU actual', zorder=5)
ax.plot(ts, df['ik_boom'].iloc[s:e],    color=C_M1,     lw=1.2, ls='--',
        label='IK reference', alpha=0.85)
ax.plot(ts, df['m4_boom'].iloc[s:e],    color=C_M4,     lw=1.2, ls=':',
        label='PID+FF sim', alpha=0.9)
ax.fill_between(ts, df['slope_boom'].iloc[s:e], df['ik_boom'].iloc[s:e],
                alpha=0.18, color=C_M1, label='Current error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title('Boom Angle (zoom-in)')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

ax = axes2[1]
ax.plot(ts, df['slope_arm'].iloc[s:e], color=C_TARGET, lw=1.5,
        label='IMU actual', zorder=5)
ax.plot(ts, df['ik_arm'].iloc[s:e],    color=C_M1,     lw=1.2, ls='--',
        label='IK reference', alpha=0.85)
ax.plot(ts, df['m4_arm'].iloc[s:e],    color=C_M4,     lw=1.2, ls=':',
        label='PID+FF sim', alpha=0.9)
ax.fill_between(ts, df['slope_arm'].iloc[s:e], df['ik_arm'].iloc[s:e],
                alpha=0.18, color=C_M1, label='Current error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title('Arm Angle (zoom-in)')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.tick_params(axis='both', labelsize=9)

plt.tight_layout()
out_zoom = f'{OUTPUT_DIR}/excavator_pid_ff_zoom.png'
plt.savefig(out_zoom, dpi=150, bbox_inches='tight')
print(f"  Saved (zoom): {out_zoom}")
