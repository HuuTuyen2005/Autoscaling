import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_metrics, load_forecast
from src.autoscaler import (
    PredictiveAutoscaler,
    CAPACITY_REQ,
)

st.set_page_config(layout="wide")
st.title("Predictive Autoscaling Demo (XGBoost-based)")

# ==============================
# Scenario definitions
# ==============================
SCENARIOS = {
    "Scale": ("1995-08-29 12:30:00-04:00", "1995-08-29 13:00:00-04:00"),
    "Stable": ("1995-08-23 00:00:00-04:00", "1995-08-23 00:20:00-04:00"),
}

scenario = st.selectbox("Select Scenario", list(SCENARIOS.keys()))
start, end = SCENARIOS[scenario]

# ==============================
# Load data
# ==============================
metrics = load_metrics()
forecast = load_forecast()

metrics = metrics.loc[start:end]
forecast = forecast.loc[metrics.index]

# ==============================
# Predictive autoscaling simulation
# ==============================
autoscaler = PredictiveAutoscaler(init_instances=1)

instances = []
capacity = []
events = []
overloads = []

for ts in metrics.index:
    r_actual = metrics.loc[ts, "requests"]
    b_actual = metrics.loc[ts, "bytes"]

    r_forecast = forecast.loc[ts, "requests_pred_5m"]
    b_forecast = forecast.loc[ts, "bytes_pred_5m"]

    result = autoscaler.step(
        req_forecast=r_forecast,
        bytes_forecast=b_forecast,
        req_actual=r_actual,
        bytes_actual=b_actual,
    )

    instances.append(result["instances"])
    capacity.append(result["instances"] * CAPACITY_REQ)
    events.append(result["event"])
    overloads.append(result["overload"])

instances = np.array(instances)
capacity = np.array(capacity)
overloads = np.array(overloads)

# ==============================
# Visualization 1: Load vs Capacity
# ==============================
fig1, ax1 = plt.subplots(figsize=(13, 4))

ax1.plot(
    metrics.index,
    metrics["requests"],
    label="Actual Requests",
    alpha=0.4,
)

ax1.plot(
    forecast.index,
    forecast["requests_pred_5m"],
    linestyle="--",
    label="Forecast (5m)",
)

ax1.step(
    metrics.index,
    capacity,
    where="post",
    label="Provisioned Capacity",
)

# Mark scaling events
for t, e, cap in zip(metrics.index, events, capacity):
    if e == "SCALE_UP":
        ax1.scatter(t, cap, marker="^")
    elif e == "SCALE_DOWN":
        ax1.scatter(t, cap, marker="v")

ax1.set_ylabel("Requests / min")
ax1.set_title("Workload vs Provisioned Capacity")
ax1.legend()
ax1.grid(alpha=0.3)

st.pyplot(fig1)

# ==============================
# Visualization 2: Instances over time
# ==============================
fig2, ax2 = plt.subplots(figsize=(13, 3))

ax2.step(
    metrics.index,
    instances,
    where="post",
    label="Active Instances",
)

ax2.set_ylabel("Instances")
ax2.set_title("Autoscaling Decisions Over Time")
ax2.legend()
ax2.grid(alpha=0.3)

st.pyplot(fig2)

# ==============================
# Visualization 3: SLA violations
# ==============================
fig3, ax3 = plt.subplots(figsize=(13, 3))

ax3.plot(
    np.cumsum(overloads),
    label="Cumulative SLA Violations",
)

ax3.set_title("Cumulative SLA Violations")
ax3.set_ylabel("Violations")
ax3.set_xlabel("Time step")
ax3.legend()
ax3.grid(alpha=0.3)

st.pyplot(fig3)

# ==============================
# Explanation box
# ==============================
st.subheader("Latest Autoscaling Decision")

last_ts = metrics.index[-1]

st.markdown(
    f"""
**Time:** {last_ts}  
**Actual load:** {metrics.iloc[-1]["requests"]:.1f} req/min  
**Forecast (5m):** {forecast.iloc[-1]["requests_pred_5m"]:.1f} req/min  
**Active instances:** {instances[-1]}  
**Provisioned capacity:** {capacity[-1]} req/min  
**Decision:** **{events[-1]}**  
**SLA violation:** {"YES" if overloads[-1] else "NO"}
"""
)

# ==============================
# Decision table
# ==============================
st.subheader("Scaling Decisions (last 10 steps)")

table = metrics[["requests", "bytes"]].copy()
table["forecast_req"] = forecast["requests_pred_5m"]
table["instances"] = instances
table["capacity_req"] = capacity
table["event"] = events
table["sla_violation"] = overloads

st.dataframe(table.tail(10))
