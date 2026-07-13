import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Wind Power LSTM Forecast",
    layout="wide"
)

st.title("Wind Power Forecasting using LSTM")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Configuration")

target_col = st.sidebar.text_input(
    "Target Column",
    value="P(t+0)"
)

seq_len = st.sidebar.number_input(
    "Sequence Length",
    value=144,
    step=1
)

train_days = st.sidebar.number_input(
    "Training Days",
    value=30,
    step=1
)

epochs = st.sidebar.number_input(
    "Epochs",
    value=80,
    step=1
)

batch_size = st.sidebar.number_input(
    "Batch Size",
    value=64,
    step=1
)

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)

# ==================================================
# RUN
# ==================================================

if uploaded_file is not None:

    try:

        with st.spinner("Loading data..."):

            df = pd.read_excel(uploaded_file)

            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            df = (
                df.sort_values("Timestamp")
                .reset_index(drop=True)
            )

        st.success("File Loaded Successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # ==================================================
        # LAST DAY
        # ==================================================

        last_timestamp = df["Timestamp"].max()

        forecast_day = last_timestamp.normalize()

        train_end = forecast_day - pd.Timedelta(minutes=10)

        train_start = (
            train_end
            - pd.Timedelta(days=train_days)
            + pd.Timedelta(minutes=10)
        )

        Train = df[
            (df["Timestamp"] >= train_start)
            &
            (df["Timestamp"] <= train_end)
        ].copy()

        Test = df[
            df["Timestamp"].dt.normalize()
            == forecast_day
        ].copy()

        st.write(
            f"Training Samples: {len(Train)}"
        )

        st.write(
            f"Testing Samples: {len(Test)}"
        )

        # ==================================================
        # NORMALIZATION
        # ==================================================

        Ptrain = Train[target_col].values

        Pmin = Ptrain.min()
        Pmax = Ptrain.max()

        def scale(x):
            return (
                (x - Pmin)
                /
                (Pmax - Pmin + 1e-8)
            )

        def descale(x):
            return (
                x * (Pmax - Pmin)
                + Pmin
            )

        PtrainN = scale(Ptrain)

        PallN = scale(
            df[target_col].values
        )

        # ==================================================
        # TRAINING DATA
        # ==================================================

        XTrain = []
        YTrain = []

        for i in range(
            len(PtrainN) - seq_len
        ):

            XTrain.append(
                PtrainN[i:i+seq_len]
            )

            YTrain.append(
                PtrainN[i+seq_len]
            )

        XTrain = np.array(XTrain)
        YTrain = np.array(YTrain)

        XTrain = XTrain.reshape(
            XTrain.shape[0],
            XTrain.shape[1],
            1
        )

        # ==================================================
        # MODEL
        # ==================================================

        with st.spinner("Training LSTM Model..."):

            model = Sequential()

            model.add(
                LSTM(
                    128,
                    input_shape=(
                        seq_len,
                        1
                    )
                )
            )

            model.add(
                Dropout(0.2)
            )

            model.add(
                Dense(1)
            )

            model.compile(
                optimizer=Adam(
                    learning_rate=0.001
                ),
                loss="mse"
            )

            model.fit(
                XTrain,
                YTrain,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )

        st.success("Training Complete")

        # ==================================================
        # FORECAST
        # ==================================================

        with st.spinner(
            "Generating Forecast..."
        ):

            predN = []

            start_idx = Test.index[0]

            for i in range(len(Test)):

                idx = start_idx + i

                hist = PallN[
                    idx-seq_len:idx
                ]

                hist = hist.reshape(
                    1,
                    seq_len,
                    1
                )

                prediction = (
                    model.predict(
                        hist,
                        verbose=0
                    )[0, 0]
                )

                predN.append(
                    prediction
                )

            predN = np.array(predN)

            pred = descale(predN)

        actual = Test[target_col].values

        # ==================================================
        # METRICS
        # ==================================================

        MAE = np.mean(
            np.abs(actual - pred)
        )

        mask = actual != 0

        MAPE = (
            np.mean(
                np.abs(
                    (
                        actual[mask]
                        - pred[mask]
                    )
                    /
                    actual[mask]
                )
            )
            * 100
        )

        R2 = 1 - (
            np.sum(
                (actual - pred) ** 2
            )
            /
            np.sum(
                (
                    actual
                    - np.mean(actual)
                ) ** 2
            )
        )

        # ==================================================
        # RESULTS TABLE
        # ==================================================

        results = pd.DataFrame(
            {
                "Timestamp":
                    Test["Timestamp"],
                "Actual":
                    actual,
                "Predicted":
                    pred
            }
        )

        st.subheader(
            f"Forecast Day: {forecast_day.date()}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "MAE",
            f"{MAE:.4f}"
        )

        c2.metric(
            "MAPE",
            f"{MAPE:.2f}%"
        )

        c3.metric(
            "R²",
            f"{R2:.4f}"
        )

        # ==================================================
        # CHART
        # ==================================================

        st.subheader(
            "Actual vs Predicted"
        )

        chart_df = (
            results.set_index(
                "Timestamp"
            )[
                [
                    "Actual",
                    "Predicted"
                ]
            ]
        )

        st.line_chart(
            chart_df
        )

        # ==================================================
        # TABLE
        # ==================================================

        st.subheader(
            "Prediction Table"
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        # ==================================================
        # DOWNLOAD
        # ==================================================

        excel_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=excel_data,
            file_name="Last_Day_Predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload an Excel file to begin."
    )
