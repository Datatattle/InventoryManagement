import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------
# Configuration and Helper Functions
# ---------------------------
inv_step = 5
fcst_step = 5
max_inventory = 100
max_fcst = 100

def discretize_value(value, step, max_val):
    return min(max_val, int(round(value / step) * step))

def get_state(inventory, forecast):
    """State: (discretized inventory, discretized forecasted demand)"""
    inv_state = discretize_value(inventory, inv_step, max_inventory)
    fcst_state = discretize_value(forecast, fcst_step, max_fcst)
    return (inv_state, fcst_state)

# Define RL action space: each action is a tuple (order_quantity, price_adjustment_factor)
order_options = [0, 5, 10, 15]
price_adjustments = [1.0, 1.10, 0.95]  # 1.0=no change; 1.10=increase by 10%; 0.95=decrease by 5%
actions = [(q, p) for q in order_options for p in price_adjustments]

def select_rl_action(Q, state):
    """Select the best action from the pre-trained Q-table for the given state.
       If state is unseen, default to (0, 1.0)."""
    if state in Q:
        action_idx = np.argmax(Q[state])
        return actions[action_idx]
    else:
        return (0, 1.0)

def train_forecast_model(df_train):
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    return model

def forecast_next_day(model, last_date):
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    # Return forecast for the day immediately after last_date
    forecast_next = forecast[forecast['ds'] > last_date].iloc[0]
    return forecast_next['yhat']

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Assume Date is in dd-mm-yyyy format; adjust if needed.
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

# ---------------------------
# Load Pre-trained Q-table (trained offline)
# ---------------------------
with open("trained_q_table.pkl", "rb") as f:
    Q = pickle.load(f)

# ---------------------------
# Streamlit App UI (V7.1)
# ---------------------------
st.title("Integrated RL-Driven Retail Inventory & Pricing Decisions (V7.1)")
st.markdown("**Fixed Set:** StoreID: S001, ProductID: P0015, Region: East")

# Load data from CSV (update the path if necessary)
data = load_data("data/inventory_data.csv")

# Define training period: all data up to November 30, 2023
train_end = pd.to_datetime("2023-11-30")
training_data = data[(data['Date'] <= train_end) &
                     (data['StoreID'] == "S001") &
                     (data['ProductID'] == "P0015") &
                     (data['Region'] == "East")].copy()

# New datapoints (simulation period): all rows after November 30, 2023
new_data = data[(data['Date'] > train_end) &
                (data['StoreID'] == "S001") &
                (data['ProductID'] == "P0015") &
                (data['Region'] == "East")].copy().sort_values("Date")

if new_data.empty:
    st.error("No new datapoints available for simulation (dates after November 30, 2023).")
    st.stop()

# Let user select a simulation date (from the available new datapoints)
sim_dates = sorted(new_data['Date'].dt.date.unique().tolist())
selected_date = st.selectbox("Select Simulation Date", sim_dates)

# Select the first row for the selected simulation date
sim_row = new_data[new_data['Date'].dt.date == selected_date].iloc[0]

# Current operational parameters come from the last historical record
current_inventory = training_data.iloc[-1]['InventoryLevel']
current_price = training_data.iloc[-1]['Price']

st.markdown(f"**Starting Inventory:** {current_inventory}")
st.markdown(f"**Starting Price:** {current_price}")

# Combine training data with new datapoints up to the day before the selected simulation date
available_data = pd.concat([training_data, new_data[new_data['Date'] < pd.to_datetime(selected_date)]])
available_data = available_data.sort_values("Date")

# Prepare data for Prophet training
df_train_prophet = available_data[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds'])

with st.spinner("Training forecasting model on available data..."):
    try:
        forecast_model = train_forecast_model(df_train_prophet)
    except Exception as e:
        st.error(f"Forecast model training failed: {e}")
        st.stop()

last_date_train = df_train_prophet['ds'].max()
forecasted_demand = forecast_next_day(forecast_model, last_date_train)

# Form state using RL: (discretized current_inventory, discretized forecasted_demand)
state = get_state(current_inventory, forecasted_demand)

# Use the pre-trained Q-table to select RL action (order quantity, price adjustment)
order_qty, price_adj = select_rl_action(Q, state)

# Apply RL action:
new_inventory = current_inventory + order_qty   # inventory after order arrives
new_price = current_price * price_adj             # adjusted price

# Get actual demand for the simulation date (from new_data)
actual_demand = sim_row['UnitsSold']

# Compute sales, ending inventory, revenue, and costs
sales = min(new_inventory, actual_demand)
ending_inventory = new_inventory - sales
revenue = sales * new_price
ordering_cost = order_qty * 5   # cost per unit ordered
holding_cost = ending_inventory * 0.1
reward = revenue - ordering_cost - holding_cost

# Calculate forecast error metrics
forecast_error = abs(actual_demand - forecasted_demand)
forecast_percentage_error = (forecast_error / actual_demand * 100) if actual_demand != 0 else 0

# ---------------------------
# Display Results Vertically (Key-Value Format)
# ---------------------------
st.header("Simulation Results for Selected Day")
st.markdown(f"**Simulation Date:** {selected_date}")
st.markdown(f"**Forecasted Demand for Next Day:** {forecasted_demand:.2f}")
st.markdown(f"**Forecast Error:** {forecast_error:.2f} units ({forecast_percentage_error:.2f}%)")
st.markdown(f"**Current Inventory:** {current_inventory}")
st.markdown(f"**RL Recommended Order Quantity:** {order_qty}")
st.markdown(f"**RL Recommended Price Adjustment Factor:** {price_adj}")
st.markdown(f"**New Price (after adjustment):** {new_price:.2f}")
st.markdown(f"**Actual Demand on Simulation Day:** {actual_demand}")
st.markdown(f"**Sales:** {sales}")
st.markdown(f"**Ending Inventory:** {ending_inventory}")
st.markdown(f"**Revenue:** {revenue:.2f}")
st.markdown(f"**Ordering Cost:** {ordering_cost}")
st.markdown(f"**Holding Cost:** {holding_cost:.2f}")
st.markdown(f"**Reward:** **{reward:.2f}**")

st.markdown("---")
st.markdown("This simulation integrates the RL layer (using a pre-trained Q-table) with forecasting, inventory optimization, and dynamic pricing. It uses new datapoints (selected by date from those after November 30, 2023) to test forecast accuracy and operational performance in a real-time–like setting.")
