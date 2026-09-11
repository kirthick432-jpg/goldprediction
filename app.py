# ============================================================
# GOLD PRICE PREDICTION
# LIVE GRADIO WEB APPLICATION
# ============================================================

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "gold_price_linear_regression.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. "
        "Please upload the trained model file."
    )

with open(MODEL_FILE, "rb") as file:
    model_data = pickle.load(file)


model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]


print("==========================================")
print("MODEL LOADED SUCCESSFULLY")
print("==========================================")
print("Features:", features)
print("Model:", type(model).__name__)


# ============================================================
# 2. DOWNLOAD LIVE YAHOO FINANCE DATA
# ============================================================

def get_live_value(ticker):
    """
    Download the latest available closing price
    from Yahoo Finance.
    """

    data = yf.download(
        ticker,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data available for {ticker}"
        )

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]

        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
    else:
        close_data = data["Close"]

    close_data = close_data.dropna()

    if len(close_data) == 0:
        raise ValueError(
            f"Close price unavailable for {ticker}"
        )

    return float(close_data.iloc[-1])


# ============================================================
# 3. GET ALL LIVE MARKET DATA
# ============================================================

def get_live_market_data():

    # USD / INR
    usd_inr = get_live_value("INR=X")

    # Silver Futures
    silver = get_live_value("SI=F")

    # Crude Oil Futures
    crude_oil = get_live_value("CL=F")

    # GLD current price for comparison
    gld = get_live_value("GLD")

    return {
        "USD_INR": usd_inr,
        "Silver_Price": silver,
        "Crude_Oil_Price": crude_oil,
        "GLD_Current": gld
    }


# ============================================================
# 4. PREDICT GOLD PRICE
# ============================================================

def predict_live_gold():

    try:

        # Get live market data
        market = get_live_market_data()

        usd_inr = market["USD_INR"]
        silver = market["Silver_Price"]
        crude = market["Crude_Oil_Price"]
        current_gld = market["GLD_Current"]

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[
                usd_inr,
                silver,
                crude
            ]],
            columns=features
        )

        # Scale input using saved scaler
        input_scaled = scaler.transform(
            input_data
        )

        # Predict
        prediction = model.predict(
            input_scaled
        )[0]

        # Difference
        difference = prediction - current_gld

        percentage_difference = (
            difference / current_gld
        ) * 100

        # Current timestamp
        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        # Result
        result = f"""
## 🥇 Gold Price Prediction

### Predicted GLD Price

# ${prediction:.2f}

---

### 📊 Live Market Data

| Market | Value |
|---|---:|
| USD/INR | ₹{usd_inr:.2f} |
| Silver | ${silver:.2f} |
| Crude Oil | ${crude:.2f} |
| Current GLD | ${current_gld:.2f} |

---

### 📈 Prediction Comparison

**Predicted GLD:** ${prediction:.2f}

**Current GLD:** ${current_gld:.2f}

**Difference:** ${difference:.2f}

**Difference %:** {percentage_difference:.2f}%

---

🕒 **Data updated:** {current_time}

⚠️ This is a machine-learning prediction for educational/project purposes,
not financial advice.
"""

        return result

    except Exception as e:

        return f"""
## ❌ Error

Unable to retrieve live market data.

**Reason:**

`{str(e)}`

Please try again after a few seconds.
"""


# ============================================================
# 5. MANUAL PREDICTION
# ============================================================

def predict_manual(
    usd_inr,
    silver_price,
    crude_oil_price
):

    try:

        # Validate inputs

        if usd_inr is None:
            return "Please enter USD/INR."

        if silver_price is None:
            return "Please enter Silver Price."

        if crude_oil_price is None:
            return "Please enter Crude Oil Price."

        # Create DataFrame
        input_data = pd.DataFrame(
            [[
                usd_inr,
                silver_price,
                crude_oil_price
            ]],
            columns=features
        )

        # Scale
        input_scaled = scaler.transform(
            input_data
        )

        # Prediction
        prediction = model.predict(
            input_scaled
        )[0]

        return f"""
## 🥇 Predicted Gold Price

# ${prediction:.2f}

### Input Values

**USD/INR:** ₹{usd_inr:.2f}

**Silver Price:** ${silver_price:.2f}

**Crude Oil Price:** ${crude_oil_price:.2f}

---

⚠️ Prediction is based on your trained Linear Regression model.
"""

    except Exception as e:

        return f"""
## ❌ Error

{str(e)}
"""


# ============================================================
# 6. GRADIO USER INTERFACE
# ============================================================

css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.title {
    text-align: center;
}

.description {
    text-align: center;
}
"""


with gr.Blocks(
    title="Gold Price Prediction",
    css=css
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🥇 Gold Price Prediction System

        ### Machine Learning Based Gold Price Prediction

        **Linear Regression + Yahoo Finance Live Market Data**
        """,
        elem_classes="title"
    )

    gr.Markdown(
        """
        This application predicts the **GLD ETF closing price**
        using USD/INR, Silver Price and Crude Oil Price.
        """,
        elem_classes="description"
    )

    # --------------------------------------------------------
    # LIVE PREDICTION
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 🔴 Live Prediction
        """
    )

    gr.Markdown(
        """
        Click the button below to download the latest available
        market values from Yahoo Finance and generate a prediction.
        """
    )

    live_button = gr.Button(
        "🔄 Get Live Data & Predict",
        variant="primary",
        size="lg"
    )

    live_output = gr.Markdown(
        value="""
        Click **Get Live Data & Predict** to start.
        """
    )

    live_button.click(
        fn=predict_live_gold,
        inputs=[],
        outputs=live_output
    )

    # --------------------------------------------------------
    # MANUAL PREDICTION
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 🧮 Manual Prediction
        """
    )

    with gr.Row():

        usd_input = gr.Number(
            label="USD / INR",
            value=83.0,
            precision=2
        )

        silver_input = gr.Number(
            label="Silver Price",
            value=25.0,
            precision=2
        )

        crude_input = gr.Number(
            label="Crude Oil Price",
            value=75.0,
            precision=2
        )

    manual_button = gr.Button(
        "Predict Gold Price",
        variant="primary"
    )

    manual_output = gr.Markdown()

    manual_button.click(
        fn=predict_manual,
        inputs=[
            usd_input,
            silver_input,
            crude_input
        ],
        outputs=manual_output
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    gr.Markdown(
        """
        ---

        ## 🤖 Model Information

        | Parameter | Value |
        |---|---|
        | Algorithm | Linear Regression |
        | Target | GLD Closing Price |
        | Feature 1 | USD/INR |
        | Feature 2 | Silver Price |
        | Feature 3 | Crude Oil Price |
        | Data Source | Yahoo Finance |
        | Model File | gold_price_linear_regression.pkl |

        ### ⚠️ Disclaimer

        This application is developed for an academic/project
        demonstration. Predictions are estimates generated by
        a machine-learning model and should not be considered
        financial advice.
        """
    )


# ============================================================
# 7. START GRADIO SERVER
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
    )
