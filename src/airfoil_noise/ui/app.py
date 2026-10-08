import os

import streamlit as st

from airfoil_noise.ui.client import (
    ApiClientError,
    api_is_available,
    request_prediction,
)

API_URL = os.getenv(
    "AIRFOIL_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Прогноз аэродинамического шума",
    page_icon="✈️",
    layout="centered",
)

st.title("Прогноз аэродинамического шума")
st.write(
    "Введите параметры авиационного профиля. "
    "Модель рассчитает ожидаемый уровень "
    "звукового давления."
)

with st.sidebar:
    st.header("Состояние сервиса")

    if api_is_available(API_URL):
        st.success("API доступен")
    else:
        st.warning("API недоступен")

    st.caption(f"Адрес API: {API_URL}")

with st.form("prediction_form"):
    frequency = st.number_input(
        "Частота, Гц",
        min_value=200.0,
        max_value=20000.0,
        value=1000.0,
        step=100.0,
    )

    angle_of_attack = st.number_input(
        "Угол атаки, градусы",
        min_value=0.0,
        max_value=22.2,
        value=5.0,
        step=0.1,
    )

    chord_length = st.number_input(
        "Длина хорды, м",
        min_value=0.0254,
        max_value=0.3048,
        value=0.15,
        step=0.001,
        format="%.4f",
    )

    free_stream_velocity = st.number_input(
        "Скорость свободного потока, м/с",
        min_value=31.7,
        max_value=71.3,
        value=55.0,
        step=0.1,
    )

    displacement_thickness = st.number_input(
        "Толщина вытеснения пограничного слоя, м",
        min_value=0.0004,
        max_value=0.0584,
        value=0.003,
        step=0.0001,
        format="%.6f",
    )

    submitted = st.form_submit_button(
        "Получить прогноз",
        type="primary",
        use_container_width=True,
    )

if submitted:
    payload = {
        "frequency": frequency,
        "angle_of_attack": angle_of_attack,
        "chord_length": chord_length,
        "free_stream_velocity": free_stream_velocity,
        "suction_side_displacement_thickness": (displacement_thickness),
    }

    try:
        result = request_prediction(
            api_url=API_URL,
            payload=payload,
        )
        prediction = float(result["predicted_scaled_sound_pressure"])
    except (
        ApiClientError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Не удалось получить прогноз. Проверьте работу API: {error}")
    else:
        st.success("Прогноз успешно рассчитан")
        st.metric(
            "Уровень звукового давления",
            f"{prediction:.2f} дБ",
        )

        with st.expander("Информация о модели"):
            st.write(f"Модель: {result['model_name']}")
            st.write(f"Версия preprocessing: {result['preprocessing_version']}")
