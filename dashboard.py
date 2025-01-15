import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Загрузка данных
@st.cache_data
def load_data(url):
    source_df = pd.read_csv(url, na_values=" ", keep_default_na=True)
    return source_df


# Предобработка данных
def preprocess_data(df):
    if df.isnull().any(axis=1).sum() > 0:
        df = df.dropna()
    if df.duplicated().sum() > 0:  # Удаление пропущенных значений
        df = df.drop_duplicates()  # Удаление дубликатов

    # Преобразование категориальных данных
    cat_columns = df.select_dtypes(["object"]).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df = del_outliers(df)

    return df


def del_outliers(df):
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print("\nВыбросы обработаны")
    return df


# Построение моделей
def train_models(df):
    x = df.drop(columns=["Life_expectancy", "Country", "Region"])
    y = df["Life_expectancy"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Линейная регрессия
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    y_pred_linear = linear_model.predict(x_test)

    # Случайный лес
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)

    return linear_model, rf_model, x, y, x_test, y_test, y_pred_linear, y_pred_rf


# Основная функция
def main():
    st.title("Анализ данных: Продолжительность жизни")

    # Загрузка данных
    st.subheader("1. Загрузка данных")
    file_path = "Life-Expectancy-Data-Averaged.csv"
    df = load_data(file_path)
    st.write("Данные:")
    st.write(df.head())

    # Общая информация о данных
    st.subheader("2. Обзор данных")
    st.write("Размер данных:", df.shape)
    missing_values = df.isnull().any(axis=1).sum()
    duplicates = df.duplicated().sum()
    st.write(f"Пропущенные значения: {missing_values}")
    st.write(f"Дубликаты: {duplicates}")

    # Предобработка данных
    df = preprocess_data(df)

    # Корреляционная матрица
    st.subheader("3. Корреляционный анализ")
    st.write("Корреляция между признаками:")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Графический анализ признаков
    st.subheader("4. Анализ ключевых признаков")
    important_features = ["Schooling", "GDP_per_capita", "Adult_mortality", "BMI"]
    for feature in important_features:
        st.write(f"Влияние {feature} на продолжительность жизни:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature, y="Life_expectancy", ax=ax)
        st.pyplot(fig)

    numerical_features = df.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    # Построение моделей
    linear_model, rf_model, x, y, x_test, y_test, y_pred_linear, y_pred_rf = (
        train_models(df)
    )

    # Важность признаков для линейной регрессии
    st.subheader("5. Важность признаков (Линейная регрессия)")
    feature_importances_linear = pd.Series(
        linear_model.coef_, index=x.columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=feature_importances_linear.values,
        y=feature_importances_linear.index,
        color="skyblue",
        ax=ax,
    )
    ax.set_title("Важность признаков (Линейная регрессия)")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признаки")
    st.pyplot(fig)

    # Важность признаков для случайного леса
    st.subheader("6. Важность признаков (Случайный лес)")
    feature_importances_rf = pd.Series(
        rf_model.feature_importances_, index=x.columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=feature_importances_rf.values,
        y=feature_importances_rf.index,
        color="skyblue",
        ax=ax,
    )
    ax.set_title("Важность признаков (Случайный лес)")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признаки")
    st.pyplot(fig)

    # Предсказание
    st.subheader("7. Предсказание")
    st.write("Введите значения признаков для предсказания:")
    user_input = {}
    for col in x_test.columns:
        user_input[col] = st.number_input(col)

    if st.button("Предсказать"):
        input_df = pd.DataFrame([user_input])
        prediction_linear = linear_model.predict(input_df)
        prediction_rf = rf_model.predict(input_df)
        st.write(
            f"Ожидаемая продолжительность жизни (Линейная регрессия): {prediction_linear[0]:.2f} лет"
        )
        st.write(
            f"Ожидаемая продолжительность жизни (Случайный лес): {prediction_rf[0]:.2f} лет"
        )


if __name__ == "__main__":
    main()
