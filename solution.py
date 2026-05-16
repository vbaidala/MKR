"""МКР з Python for Data Science — наскрізний кейс «Метеослужба».

Перед запуском скрипта підніміть персональний Docker-контейнер з MySQL:

    docker pull asterindex/pfds-mkr-g2-01:latest
    docker run -d -p 3306:3306 --name mkr asterindex/pfds-mkr-g2-01:latest

Потім зачекайте приблизно 30 секунд на ініціалізацію MySQL і запускайте:

    python solution.py

Графіки зберігаються в підпапку `plots/` поряд зі скриптом.
"""

# ====================================================================
# Прізвище, ім'я, по батькові: Байдала Віра Юріївна
# Група: ЗК-32
# Дата виконання:16.05.2026
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import InterfaceError, OperationalError

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    """підключитися до mysql і завантажити таблицю observations"""
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)

    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except (OperationalError, InterfaceError):
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})...")
            time.sleep(delay)

    raise RuntimeError("Не вдалося завантажити дані")


# ====================================================================
# БЛОК 1. NumPy
# ====================================================================

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    # отримуємо сирі колонки як numpy-масиви
    temperature = df_raw["temperature_c"].to_numpy(dtype=float)
    humidity = df_raw["humidity_pct"].to_numpy(dtype=float)
    wind_speed = df_raw["wind_speed_ms"].to_numpy(dtype=float)
    obs_ids = df_raw["obs_id"].to_numpy()
    datetimes = df_raw["datetime"].to_numpy()

    # будуємо apparent temperature за формулою з умови
    apparent = temperature - (100 - humidity) / 5
    print(
        f"1) T_app: len={len(apparent)}, "
        f"min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}"
    )

    # замінюємо фізичні викиди на np.nan
    temperature_outlier_mask = (temperature > 60) | (temperature < -60)
    wind_outlier_mask = wind_speed > 100

    temperature_clean = np.where(temperature_outlier_mask, np.nan, temperature)
    wind_clean = np.where(wind_outlier_mask, np.nan, wind_speed)

    print(f"2) Викидів температури замінено: {np.sum(temperature_outlier_mask)}")
    print(f"   Викидів вітру замінено:       {np.sum(wind_outlier_mask)}")
    print(f"   Перевірка wind_clean: nan={np.isnan(wind_clean).sum()}")

    # рахуємо mean / median / std вручну, ігноруючи nan
    valid_temperature = temperature_clean[~np.isnan(temperature_clean)]
    mean_t = np.nansum(valid_temperature) / valid_temperature.size
    median_t = np.nanmedian(temperature_clean)
    std_t = np.sqrt(np.nansum((valid_temperature - mean_t) ** 2) / valid_temperature.size)

    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # рахуємо кількість морозних і жарких спостережень через маски
    n_frost = np.sum(temperature_clean < 0)
    n_hot = np.sum(temperature_clean > 30)
    print(f"4) морозних: {n_frost}    жарких: {n_hot}")

    # знаходимо argmax / argmin температури без nan
    max_idx = np.nanargmax(temperature_clean)
    min_idx = np.nanargmin(temperature_clean)

    print("5) Максимальна температура:")
    print(
        f"   obs_id={obs_ids[max_idx]}, datetime={datetimes[max_idx]}, "
        f"temperature={temperature_clean[max_idx]:.2f}"
    )

    print("   Мінімальна температура:")
    print(
        f"   obs_id={obs_ids[min_idx]}, datetime={datetimes[min_idx]}, "
        f"temperature={temperature_clean[min_idx]:.2f}"
    )


# ====================================================================
# БЛОК 2. Pandas — очищення
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # перевіряємо типи колонок і базову статистику
    print("1) Типи колонок:")
    df.info()

    print("\n   Опис числових колонок:")
    print(df.describe().round(2).to_string())

    # переводимо datetime у тип datetime і встановлюємо як індекс
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # видаляємо повні дублі рядків
    rows_before_duplicates = len(df)
    df = df.drop_duplicates()
    n_dups = rows_before_duplicates - len(df)
    print(f"2) drop_duplicates: видалено {n_dups}")

    # заповнюємо humidity_pct медіаною по місяцю в межах міста
    humidity_missing_before = df["humidity_pct"].isna().sum()

    df["month"] = df.index.month
    df["humidity_pct"] = df.groupby(["city", "month"])["humidity_pct"].transform(
        lambda s: s.fillna(s.median())
    )

    humidity_missing_after = df["humidity_pct"].isna().sum()
    n_filled = humidity_missing_before - humidity_missing_after
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # прибираємо фізичні викиди температури і швидкості вітру
    rows_before_outliers = len(df)

    temperature_mask = df["temperature_c"].between(-60, 60)
    wind_mask = df["wind_speed_ms"].isna() | df["wind_speed_ms"].between(0, 60)

    df = df[temperature_mask & wind_mask].copy()
    n_outliers = rows_before_outliers - len(df)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    # видаляємо допоміжну колонку
    df = df.drop(columns=["month"])

    # виводимо звіт очищення
    print(f"\n   Звіт: {rows_before} → {len(df)} рядків")
    print(f"   Видалено дублів: {n_dups}")
    print(f"   Заповнено NaN humidity_pct: {n_filled}")
    print(f"   Видалено викидів: {n_outliers}")

    return df


# ====================================================================
# БЛОК 3. Pandas — аналітика
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # середня температура по містах
    by_city_temp = df.groupby("city")["temperature_c"].mean().sort_values(ascending=False)
    warmest_city = by_city_temp.idxmax()
    coldest_city = by_city_temp.idxmin()

    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())
    print(f"   Найтепліше місто: {warmest_city}")
    print(f"   Найхолодніше місто: {coldest_city}")

    # сумарні опади по містах
    by_city_precip = df.groupby("city")["precipitation_mm"].sum().sort_values(ascending=False)
    wettest_city = by_city_precip.idxmax()

    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())
    print(f"   Найвологіше місто: {wettest_city}")

    # місячна середня температура
    try:
        monthly_mean = df["temperature_c"].resample("ME").mean()
    except ValueError:
        monthly_mean = df["temperature_c"].resample("M").mean()

    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # pivot: місто × місяць, значення = середня температура
    pivot = df.pivot_table(
        values="temperature_c",
        index="city",
        columns=df.index.month,
        aggfunc="mean",
    ).sort_index(axis=1)

    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # кількість днів з опадами > 5 мм по містах
    daily_precip = (
        df.groupby("city")
        .resample("D")["precipitation_mm"]
        .sum(min_count=1)
        .reset_index()
    )

    rainy_days = (
        daily_precip.groupby("city")["precipitation_mm"]
        .apply(lambda s: int((s > 5).sum()))
        .sort_values(ascending=False)
    )

    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # знаходимо аномальний місяць через відхилення від норми календарного місяця
    monthly_norm = monthly_mean.groupby(monthly_mean.index.month).mean()
    deviations = monthly_mean.copy()

    for idx in deviations.index:
        deviations.loc[idx] = monthly_mean.loc[idx] - monthly_norm.loc[idx.month]

    anomaly_idx = deviations.abs().idxmax()
    anomaly_month = (anomaly_idx.year, anomaly_idx.month)
    anomaly_dev = deviations.loc[anomaly_idx]
    anomaly_type = "хвиля спеки" if anomaly_dev > 0 else "холодна хвиля"

    print(
        f"\n6) Аномальний місяць: {anomaly_month}  "
        f"відхилення = {anomaly_dev:+.2f}°C"
    )
    print(f"   Тип аномалії: {anomaly_type}")

    # визначаємо стабільніший регіон за стандартним відхиленням температури
    region_std = df.groupby("region")["temperature_c"].std().sort_values()
    most_stable_region = region_std.idxmin()

    print("\n   Стандартне відхилення температури по регіонах:")
    print(region_std.round(2).to_string())
    print(f"   Найстабільніший регіон: {most_stable_region}")

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
        "rainy_days": rainy_days,
        "anomaly_month": anomaly_month,
        "anomaly_dev": anomaly_dev,
        "anomaly_type": anomaly_type,
        "warmest_city": warmest_city,
        "coldest_city": coldest_city,
        "wettest_city": wettest_city,
        "region_std": region_std,
        "most_stable_region": most_stable_region,
    }


# ====================================================================
# БЛОК 4. Matplotlib + інтерпретація
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # графік 1: місячна динаміка температури по 3 містах
    selected_cities = sorted(df["city"].dropna().unique())[:3]

    fig, ax = plt.subplots(figsize=(11, 5))

    for city in selected_cities:
        city_df = df[df["city"] == city]
        try:
            city_monthly = city_df["temperature_c"].resample("ME").mean()
        except ValueError:
            city_monthly = city_df["temperature_c"].resample("M").mean()

        ax.plot(city_monthly.index, city_monthly.values, marker="o", label=city)

    ax.set_title("Місячна динаміка температури для 3 міст")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Середня температура, °C")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # графік 2: сумарні опади по містах
    fig, ax = plt.subplots(figsize=(8, 5))

    analytics["by_city_precip"].plot(kind="bar", ax=ax)

    ax.set_title("Сумарні опади по містах")
    ax.set_xlabel("Місто")
    ax.set_ylabel("Опади, мм")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # графік 3: розподіл температур з mean і median
    fig, ax = plt.subplots(figsize=(9, 5))

    temp_values = df["temperature_c"].dropna()
    mean_temp = temp_values.mean()
    median_temp = temp_values.median()

    ax.hist(temp_values, bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(mean_temp, linestyle="--", linewidth=2, label=f"mean = {mean_temp:.2f}")
    ax.axvline(median_temp, linestyle=":", linewidth=2, label=f"median = {median_temp:.2f}")

    ax.set_title("Розподіл температури")
    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Кількість спостережень")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # графік 4: heatmap місто × місяць
    fig, ax = plt.subplots(figsize=(11, 5))

    pivot = analytics["pivot"]
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Heatmap середньої температури: місто × місяць")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Місто")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    fig.colorbar(image, ax=ax, label="Середня температура, °C")

    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"4 графіки збережені в {PLOTS_DIR}/")


def print_generated_conclusions(analytics: dict) -> None:
    section("АВТОМАТИЧНО ЗГЕНЕРОВАНІ ВИСНОВКИ")

    anomaly_year, anomaly_month = analytics["anomaly_month"]
    anomaly_dev = analytics["anomaly_dev"]

    conclusions = (
        f"За результатами аналізу найтеплішим містом є {analytics['warmest_city']}, "
        f"а найхолоднішим містом є {analytics['coldest_city']}. "
        f"Сезонність температури виражена чітко: у місячній динаміці видно "
        f"зростання температур у теплий період року та зниження у холодний період. "
        f"Найвологішим містом за сумарною кількістю опадів є {analytics['wettest_city']}. "
        f"Аномальним місяцем визначено {anomaly_month:02d}.{anomaly_year}, "
        f"оскільки його середня температура найбільше відхиляється від кліматичної "
        f"норми для відповідного календарного місяця; відхилення становить "
        f"{anomaly_dev:+.2f} °C, тому це {analytics['anomaly_type']}. "
        f"Найстабільнішим за температурою регіоном є {analytics['most_stable_region']}, "
        f"бо для нього отримано найменше стандартне відхилення температури. "
        f"Для практичних рішень варто враховувати міста з найбільшою кількістю "
        f"опадів під час планування інфраструктури, а температурні аномалії — "
        f"під час підготовки сезонних кліматичних звітів."
    )

    print(conclusions)


# ====================================================================

def main() -> None:
    df_raw = load_observations()
    print(f"Завантажено: shape={df_raw.shape}")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)
    print_generated_conclusions(analytics)


if __name__ == "__main__":
    main()


"""
ВИСНОВКИ.

У результаті аналізу метеоданих встановлено, що найтеплішим містом є Харків
із середньою температурою 12.66 °C, а найхолоднішим містом є Одеса із
середньою температурою 8.13 °C. Сезонність температури виражена чітко:
температура зростає від зими до літа та знижується восени й узимку.
Найвологішим містом є Львів, оскільки він має найбільшу сумарну кількість
опадів — 495.7 мм, а також найбільше днів з опадами понад 5 мм — 20 днів.
Аномальним місяцем визначено липень 2023 року, бо його середня температура
найбільше відхиляється від кліматичної норми для липня. Відхилення становить
+3.74 °C, тому це була хвиля спеки. Найстабільнішим за температурою регіоном
є East, оскільки для нього отримано найменше стандартне відхилення
температури. Перед аналізом було видалено 233 дублікати, заповнено
584 пропущені значення вологості та прибрано 172 фізичні викиди, що дозволило
отримати коректніші результати. Практично ці результати можна використати
для планування інфраструктури у містах з більшою кількістю опадів і для
підготовки до літніх хвиль спеки.
"""
