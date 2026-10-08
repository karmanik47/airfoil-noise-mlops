import os

import streamlit as st

from airfoil_noise.ui.client import (
    ApiClientError,
    request_prediction,
)

API_URL = os.getenv(
    "AIRFOIL_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Прогноз аэродинамического шума",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #17191d;
            color: #ececec;
        }

        [data-testid="stHeader"] {
            background-color: #17191d;
        }

        .block-container {
            max-width: 900px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }

        h1 {
            color: #f2f2f2;
            font-size: 2.7rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem !important;
        }

        .description {
            color: #b7bac0;
            font-size: 1.15rem;
            line-height: 1.6;
            margin-bottom: 2rem;
        }

        .form-title {
            color: #dedede;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        div[data-testid="stForm"] {
            background-color: #21242a;
            border: 1px solid #353941;
            border-radius: 6px;
            padding: 1.8rem;
        }

        div[data-testid="stForm"] label p {
            color: #dedede;
            font-size: 1.08rem !important;
            font-weight: 500;
        }

        div[data-baseweb="input"] {
            background-color: #191b20;
        }

        div[data-baseweb="input"] input {
            color: #f0f0f0;
            font-size: 1.08rem;
        }

        div[data-testid="stFormSubmitButton"] {
            margin-top: 0.8rem;
        }

        div[data-testid="stFormSubmitButton"] button {
            background-color: #c99b57;
            color: #181818;
            border: none;
            border-radius: 4px;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.6rem 1.6rem;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #d8aa64;
            color: #181818;
        }

        .result {
            background-color: #21242a;
            border: 1px solid #353941;
            border-top: 3px solid #c99b57;
            border-radius: 4px;
            padding: 1.5rem 1.7rem;
            margin-top: 2rem;
        }

        .result-label {
            color: #b7bac0;
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
        }

        .result-value {
            color: #f2f2f2;
            font-size: 2.2rem;
            font-weight: 600;
        }

        .result-details {
            color: #92969e;
            font-size: 0.95rem;
            margin-top: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Прогноз аэродинамического шума")

st.markdown(
    """
    <div class="description">
        Введите параметры авиационного профиля,
        чтобы рассчитать ожидаемый уровень
        звукового давления.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="form-title">Параметры профиля</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    left_column, right_column = st.columns(2)

    with left_column:
        frequency = st.number_input(
            "Частота, Гц",
            min_value=200.0,
            max_value=20000.0,
            value=1000.0,
            step=100.0,
        )

        chord_length = st.number_input(
            "Длина хорды, м",
            min_value=0.0254,
            max_value=0.3048,
            value=0.15,
            step=0.001,
            format="%.4f",
        )

        displacement_thickness = st.number_input(
            "Толщина пограничного слоя, м",
            min_value=0.0004,
            max_value=0.0584,
            value=0.003,
            step=0.0001,
            format="%.6f",
        )

    with right_column:
        angle_of_attack = st.number_input(
            "Угол атаки, °",
            min_value=0.0,
            max_value=22.2,
            value=5.0,
            step=0.1,
        )

        free_stream_velocity = st.number_input(
            "Скорость потока, м/с",
            min_value=31.7,
            max_value=71.3,
            value=55.0,
            step=0.1,
        )

    submitted = st.form_submit_button("Рассчитать")

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
    ):
        st.error("Не удалось получить прогноз. Проверьте, запущен ли API.")
    else:
        st.markdown(
            f"""
            <div class="result">
                <div class="result-label">
                    Прогнозируемый уровень
                    звукового давления
                </div>
                <div class="result-value">
                    {prediction:.2f} дБ
                </div>
                <div class="result-details">
                    Модель: {result["model_name"]};
                    обработка данных:
                    {result["preprocessing_version"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
