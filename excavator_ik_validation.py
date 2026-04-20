"""
KOCETI 30ton Excavator - Path Error Analysis
============================================
실험 흐름:
  1. FK 검증: IMU 관절각 → FK → 버킷 위치 vs 실측 버킷 위치
  2. 방법1 (IK 정밀도 오차):
       목표경로 → IK → 기준관절각 → FK → 복원경로
       복원경로 vs 목표경로 [mm]
       → IK가 완벽히 실행됐을 때의 이론적 한계 오차
       → feedforward로 줄일 수 있음
  3. 방법2 (현재 경로 오차):
       실제 IMU 관절각 → FK → 실제경로
       실제경로 vs 목표경로 [mm]
       → 현재 굴착기가 제어 없이 움직일 때의 오차
       → PID로 줄여야 할 오차
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

DATA_PATH  = 'data/meta_1046.csv'
OUTPUT_DIR = '.'

# 링크 파라미터 [mm]
a2 = 6245   # Boom
a3 = 3113   # Arm
a4 = 1910   # Bucket

# 캘리브레이션 파라미터 (데이터 기반 최적화 추정값)
BOOM_OFF  = -1.887   # deg
ARM_OFF   =  0.334   # deg
BKT_OFF   =  0.478   # deg
REACH_OFF =  100.6   # mm
Z_OFF     =  49.5    # mm

DT = 0.1  # 10Hz

# ── 1. 기구학 함수 ───────────────────────────────────────────────────────────

def fk(boom_deg, arm_deg, bkt_deg):
    """
    순기구학 (FK): 관절각 → 버킷 끝점 위치 (시상면)
    출력: reach [mm], z [mm]
    """
    b = np.deg2rad(boom_deg)
    a = np.deg2rad(arm_deg)
    k = np.deg2rad(bkt_deg)
    reach = a2*np.cos(b) + a3*np.cos(b+a) + a4*np.cos(b+a+k)
    z     = a2*np.sin(b) + a3*np.sin(b+a) + a4*np.sin(b+a+k)
    return reach, z


def fk_cal(boom_deg, arm_deg, bkt_deg):
    """캘리브레이션 오프셋 적용 FK"""
    b = np.deg2rad(boom_deg + BOOM_OFF)
    a = np.deg2rad(arm_deg  + ARM_OFF)
    k = np.deg2rad(bkt_deg  + BKT_OFF)
    reach = a2*np.cos(b) + a3*np.cos(b+a) + a4*np.cos(b+a+k) + REACH_OFF
    z     = a2*np.sin(b) + a3*np.sin(b+a) + a4*np.sin(b+a+k) + Z_OFF
    return reach, z


def ik(reach, z, boom_deg, arm_deg, bkt_deg):
    """
    역기구학 (IK): 버킷 끝점 위치 → 관절각
    출력: (boom_ref, arm_ref) [deg], 도달 불가 시 (None, None)
    주의: bucket 절대각 = boom+arm+bucket 사용 (FK와 일관성 유지)
    """
    bkt_abs = np.deg2rad(boom_deg + arm_deg + bkt_deg)
    xw = reach - a4 * np.cos(bkt_abs)
    zw = z     - a4 * np.sin(bkt_abs)
    D  = (xw**2 + zw**2 - a2**2 - a3**2) / (2 * a2 * a3)
    if abs(D) > 1.0:
        return None, None
    arm  = np.degrees(np.arctan2(-np.sqrt(max(0, 1 - D**2)), D))
    boom = np.degrees(
        np.arctan2(zw, xw) -
        np.arctan2(a3 * np.sin(np.deg2rad(arm)),
                   a2 + a3 * np.cos(np.deg2rad(arm))))
    return boom, arm


# ── 2. 데이터 로드 ───────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH, skipinitialspace=True)
N  = len(df)
t  = np.arange(N) * DT

# 목표 경로: CSV 실측 버킷 궤적 (swing 제거 → 시상면 기준)
df['ref_reach'] = np.sqrt(df['bucket_x']**2 + df['bucket_y']**2)
df['ref_z']     = df['bucket_z']

print(f"  Samples : {N},  Duration: {t[-1]:.0f}s")
print(f"  Boom    : {df['slope_boom'].min():.1f} ~ {df['slope_boom'].max():.1f} deg")
print(f"  Arm     : {df['slope_arm'].min():.1f} ~ {df['slope_arm'].max():.1f} deg")
print(f"  Bucket  : {df['slope_bucket'].min():.1f} ~ {df['slope_bucket'].max():.1f} deg")


# ── 3. FK 검증 ───────────────────────────────────────────────────────────────
# IMU 관절각 → FK(calibrated) → 버킷 위치 vs 실측 버킷 위치

print("\n[FK Validation]")

df['fk_cal_reach'], df['fk_cal_z'] = zip(*df.apply(
    lambda r: fk_cal(r['slope_boom'], r['slope_arm'], r['slope_bucket']), axis=1))

df['fk_cal_err'] = np.sqrt(
    (df['fk_cal_reach'] - df['ref_reach'])**2 +
    (df['fk_cal_z']     - df['ref_z']    )**2)

fk_rmse = np.sqrt((df['fk_cal_err']**2).mean())
print(f"  FK 2D RMSE (calibrated): {fk_rmse:.1f} mm")


# ── 4. 방법1: IK 정밀도 오차 ────────────────────────────────────────────────
# 목표경로 → IK → 기준관절각 → FK → 복원경로 vs 목표경로

print("\n[Method1: IK Precision Error]")
print("  target path -> IK -> FK -> restored path vs target path")

ik_res = df.apply(lambda r: ik(
    r['ref_reach'], r['ref_z'],
    r['slope_boom'], r['slope_arm'], r['slope_bucket']), axis=1)

df['m1_boom'] = [r[0] for r in ik_res]
df['m1_arm']  = [r[1] for r in ik_res]

df['m1_reach'], df['m1_z'] = zip(*df.apply(
    lambda r: fk(r['m1_boom'], r['m1_arm'], r['slope_bucket'])
    if r['m1_boom'] is not None else (np.nan, np.nan), axis=1))

df['err_m1'] = np.sqrt(
    (df['m1_reach'] - df['ref_reach'])**2 +
    (df['m1_z']     - df['ref_z']    )**2)

rmse_m1 = np.sqrt((df['err_m1']**2).mean())
print(f"  RMSE: {rmse_m1:.1f} mm  (IK 정밀도 한계, feedforward로 줄일 수 있음)")


# ── 5. 방법2: 현재 경로 오차 ────────────────────────────────────────────────
# 실제 IMU 관절각 → FK → 실제경로 vs 목표경로

print("\n[Method2: Actual Path Error]")
print("  actual IMU angles -> FK -> actual path vs target path")

df['m2_reach'], df['m2_z'] = zip(*df.apply(
    lambda r: fk(r['slope_boom'], r['slope_arm'], r['slope_bucket']), axis=1))

df['err_m2'] = np.sqrt(
    (df['m2_reach'] - df['ref_reach'])**2 +
    (df['m2_z']     - df['ref_z']    )**2)

rmse_m2 = np.sqrt((df['err_m2']**2).mean())
print(f"  RMSE: {rmse_m2:.1f} mm  (현재 오차, PID로 줄여야 할 양)")


# ── 6. 결과 요약 ─────────────────────────────────────────────────────────────

print("\n" + "="*52)
print("  RESULTS SUMMARY")
print("="*52)
print(f"  FK  (calibrated)          : {fk_rmse:.1f} mm")
print(f"  Method1 (IK precision)    : {rmse_m1:.1f} mm  <- feedforward 목표")
print(f"  Method2 (current error)   : {rmse_m2:.1f} mm  <- PID 적용 전")
print(f"  Gap (PID needs to close)  : {rmse_m2 - rmse_m1:.1f} mm")
print("="*52)


# ── 7. 그래프 ────────────────────────────────────────────────────────────────

print("\n[Plotting...]")

matplotlib.rcParams['axes.unicode_minus'] = False

C_TARGET = '#444441'
C_M1     = '#185FA5'
C_M2     = '#993C1D'
C_IK     = '#0F6E56'

# 관절각 오차 계산 (IMU vs IK 기준각)
df['err_boom_deg'] = df['m1_boom'] - df['slope_boom']
df['err_arm_deg']  = df['m1_arm']  - df['slope_arm']
rmse_boom = np.sqrt((df['err_boom_deg']**2).mean())
rmse_arm  = np.sqrt((df['err_arm_deg']**2).mean())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 그래프1. 시상면 궤적 비교 ────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(df['ref_reach']/1000, df['ref_z']/1000,
        color=C_TARGET, lw=2.0, label='Target path', zorder=5)
ax.plot(df['m1_reach']/1000, df['m1_z']/1000,
        color=C_M1, lw=1.3, ls='--',
        label=f'SW verification  RMSE={rmse_m1:.0f} mm', alpha=0.9)
ax.plot(df['m2_reach']/1000, df['m2_z']/1000,
        color=C_M2, lw=1.3, ls=':',
        label=f'HW verification  RMSE={rmse_m2:.0f} mm', alpha=0.9)
ax.set_xlabel('Reach [m]'); ax.set_ylabel('Z [m]')
ax.set_title('1. Sagittal Plane Trajectory')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ── 그래프2. 경로 오차 시계열 ────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(t, df['err_m1'], color=C_M1, lw=0.9,
        label=f'SW verification  RMSE={rmse_m1:.0f} mm', alpha=0.85)
ax.plot(t, df['err_m2'], color=C_M2, lw=0.9,
        label=f'HW verification  RMSE={rmse_m2:.0f} mm', alpha=0.85)
ax.axhline(rmse_m1, color=C_M1, ls='--', lw=1.2, alpha=0.6)
ax.axhline(rmse_m2, color=C_M2, ls='--', lw=1.2, alpha=0.6)
ax.fill_between(t, df['err_m1'], df['err_m2'],
                alpha=0.10, color='orange',
                label=f'Gap (PID target): {rmse_m2 - rmse_m1:.0f} mm')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Path error [mm]')
ax.set_title('2. Path Following Error over Time')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ── 그래프3. Boom 관절각: IMU vs IK 기준각 ───────────────────────────────
ax = axes[1, 0]
ax.plot(t, df['slope_boom'], color=C_TARGET, lw=1.2,
        label='IMU actual', zorder=5)
ax.plot(t, df['m1_boom'], color=C_IK, lw=1.0, ls='--',
        label=f'IK reference  RMSE={rmse_boom:.2f} deg', alpha=0.85)
ax.fill_between(t, df['slope_boom'], df['m1_boom'],
                alpha=0.15, color=C_IK, label='Error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title(f'3. Boom Angle: IMU vs IK reference  (RMSE={rmse_boom:.2f} deg)')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ── 그래프4. Arm 관절각: IMU vs IK 기준각 ────────────────────────────────
ax = axes[1, 1]
ax.plot(t, df['slope_arm'], color=C_TARGET, lw=1.2,
        label='IMU actual', zorder=5)
ax.plot(t, df['m1_arm'], color=C_M2, lw=1.0, ls='--',
        label=f'IK reference  RMSE={rmse_arm:.2f} deg', alpha=0.85)
ax.fill_between(t, df['slope_arm'], df['m1_arm'],
                alpha=0.15, color=C_M2, label='Error band')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
ax.set_title(f'4. Arm Angle: IMU vs IK reference  (RMSE={rmse_arm:.2f} deg)')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.tight_layout()
out_path = f'{OUTPUT_DIR}/excavator_ik_validation.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")
