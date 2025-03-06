import numpy as np
import matplotlib.pyplot as plt

# Assignment1
M_p = 938272046000 # milli eV
M_e = 510998926 # milli eV
M_ratio = M_p / M_e # 상대 비율

comparison = 1836.12 # reference 6*(pi**5) on Phys rev B 82 554

print(f"Mass Ratio: {M_ratio}")
# Mass Ratio: 1836.1526771584643

relatie_error = 100 * (M_ratio - comparison) / comparison

print(f"Relatie_error: {relatie_error} %")
# Relatie_error: 0.0017796853399759595 %


# Assignment 2

v0_kmh = 100 # km/h
g = 10 # m/s^2
theta_deg = 45 # 발사 각도  45˚

v0_ms = v0_kmh * (1000 / 3600)  # m/s
theta = np.radians(theta_deg)  # radian

# TOF (Time of Flight)
T = (2 * v0_ms * np.sin(theta)) / g

# time vector
t = np.linspace(0, T, num=1000)

# Motion Equation
x = (v0_ms * np.cos(theta)) * t
y = (v0_ms * np.sin(theta)) * t - (g * t**2 / 2)

# Trajectory
plt.figure(figsize=(16, 5))
plt.plot(x, y, label='Projectile Motion')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Motion Trajectory')
plt.axhline(0, color='black', linewidth=0.1)
plt.legend()
plt.grid()
plt.savefig("Trajectory.png", dpi=300)
plt.show()
plt.close()


# Assignment 3
from datetime import datetime

filenames = ["seoul_temp.txt", "seoul_temp1.txt"]

plt.figure(figsize=(15, 5))  # 그래프 크기 설정

dates, avg_values = [], []  # 날짜 및 평균값

for i, filename in enumerate(filenames):
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()  # 공백 기준 분리
            if len(parts) == 4:  # 데이터 유효성 검사
                # 날짜 변환
                date_obj = datetime.strptime(parts[0], "%Y-%m-%d")
                dates.append(np.datetime64(date_obj))
                avg_values.append(float(parts[2]))  # 두 번째 값 사용

    arr_dates = np.array(dates)
    arr_avg = np.array(avg_values)
    
    plt.plot(arr_dates, arr_avg, marker="o", linestyle="None", color="red")

# 그래프 설정
plt.xlabel("Date")
plt.ylabel("Average Tempreature (˚C)")
plt.title("1908-2021 Seoul Average Temperature")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("Seoul Average Temperature (1908-2021).png", dpi=300)
plt.show()
plt.close()

