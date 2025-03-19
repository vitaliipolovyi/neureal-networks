# %%
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# %%
# Задача: від якості обслуговування та якості їжі до суми чайових
x_service = np.arange(0, 10.01, 0.5)
x_food = np.arange(0, 10.01, 0.5)
x_tip = np.arange(0, 25.01, 1.0)

# %%
# Функції належності
service_low = fuzz.trimf(x_service, [0, 0, 5])
service_middle = fuzz.trimf(x_service, [0, 5, 10])
service_high = fuzz.trimf(x_service, [5, 10, 10])

food_low = fuzz.zmf(x_food, 0, 5)
food_middle = fuzz.pimf(x_food, 0, 4, 5, 10)
food_high = fuzz.smf(x_food, 5, 10)

tip_low = fuzz.trimf(x_tip, [0, 0, 13])
tip_middle = fuzz.trimf(x_tip, [0, 13, 25])
tip_high = fuzz.trimf(x_tip, [13, 25, 25])

# %%
# Вхідні дані: оцінка за обслуговування та оцінка за їжу
service_score = 1.0
food_score = 1.0

service_low_degree = fuzz.interp_membership(
    x_service, service_low, service_score
)
service_middle_degree = fuzz.interp_membership(
    x_service, service_middle, service_score
)
service_high_degree = fuzz.interp_membership(
    x_service, service_high, service_score
)

food_low_degree = fuzz.interp_membership(x_food, food_low, food_score)
food_middle_degree = fuzz.interp_membership(x_food, food_middle, food_score)
food_high_degree = fuzz.interp_membership(x_food, food_high, food_score)

# %%
fig_scale_x = 2.0
fig_scale_y = 1.5
fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
row = 2
col = 3

plt.subplot(row, col, 1)
plt.title("Якість послуг")
plt.plot(x_service, service_low, label="low", marker=".")
plt.plot(x_service, service_middle, label="middle", marker=".")
plt.plot(x_service, service_high, label="high", marker=".")
plt.plot(service_score, 0.0, label="service score", marker="D")
plt.plot(service_score, service_low_degree,
         label="low degree", marker="o")
plt.plot(service_score, service_middle_degree,
         label="middle degree", marker="o")
plt.plot(service_score, service_high_degree,
         label="high degree", marker="o")
plt.legend(loc="upper left")

plt.subplot(row, col, 2)
plt.title("Якість їжі")
plt.plot(x_food, food_low, label="low", marker=".")
plt.plot(x_food, food_middle, label="middle", marker=".")
plt.plot(x_food, food_high, label="high", marker=".")
plt.plot(food_score, 0.0, label="food score", marker="D")
plt.plot(food_score, food_low_degree, label="low degree", marker="o")
plt.plot(food_score, food_middle_degree, label="middle degree", marker="o")
plt.plot(food_score, food_high_degree, label="high degree", marker="o")
plt.legend(loc="upper left")

plt.subplot(row, col, 3)
plt.title("Чайові")
plt.plot(x_tip, tip_low, label="low", marker=".")
plt.plot(x_tip, tip_middle, label="middle", marker=".")
plt.plot(x_tip, tip_high, label="high", marker=".")
plt.legend(loc="upper left")

# =======================================
# погані їжа або погані послуги
low_degree = np.fmax(service_low_degree, food_low_degree)
# посередні послуги
middle_degree = service_middle_degree
# добра їжа або якісні послуги
high_degree = np.fmax(service_high_degree, food_high_degree)

plt.subplot(row, col, 4)
plt.title("Деякі нечіткі правила")
t = ("пог. їжа або пог. послуги <-> погано\n"
     "пос. послуги <-> посередньо\n"
     "добр. їжа або якіс. послуги <-> добре")
plt.text(0.1, 0.5, t)

activation_low = np.fmin(low_degree, tip_low)
activation_middle = np.fmin(middle_degree, tip_middle)
activation_high = np.fmin(high_degree, tip_high)

plt.subplot(row, col, 5)
plt.title("Активація Мамдані")

plt.plot(x_tip, activation_low, label="low tip", marker=".")
plt.plot(x_tip, activation_middle, label="middle tip", marker=".")
plt.plot(x_tip, activation_high, label="high tip", marker=".")
plt.legend(loc="upper left")

# Застосування правил:
aggregated = np.fmax(
    activation_low,
    np.fmax(activation_middle, activation_high)
)

# Дефазифікація
tip_centroid = fuzz.defuzz(x_tip, aggregated, 'centroid')
tip_bisector = fuzz.defuzz(x_tip, aggregated, 'bisector')
tip_mom = fuzz.defuzz(x_tip, aggregated, "mom")
tip_som = fuzz.defuzz(x_tip, aggregated, "som")
tip_lom = fuzz.defuzz(x_tip, aggregated, "lom")

print(tip_centroid)
print(tip_bisector)
print(tip_mom)
print(tip_som)
print(tip_lom)

plt.subplot(row, col, 6)
plt.title("Агрегація та Дефазифікація")
plt.plot(x_tip, aggregated, label="fuzzy result", marker=".")
plt.plot(tip_centroid, 0.0, label="centroid", marker="o")
plt.plot(tip_bisector, 0.0, label="bisector", marker="o")
plt.plot(tip_mom, 0.0, label="mom", marker="o")
plt.plot(tip_som, 0.0, label="som", marker="o")
plt.plot(tip_lom, 0.0, label="lom", marker="o")
plt.legend(loc="upper left")

plt.show()