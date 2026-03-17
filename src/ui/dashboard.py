import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time

# Page config
st.set_page_config(page_title="OpKAN Telemetry Dashboard", layout="wide")

st.title("🛡️ OpKAN: Physics-Informed KAN Telemetry")
st.markdown("### Real-time Monitoring of Heston PDE Solver & LiuClaw Agent")

# Sidebar for status and config
st.sidebar.header("System Status")
status_placeholder = st.sidebar.empty()
regime_placeholder = st.sidebar.empty()

# Persistent state for mock demonstration if no real log exists
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
    st.session_state.loss_history = pd.DataFrame(columns=['epoch', 'pde_loss', 'bnd_loss'])
    st.session_state.mutations = []

# Mock data generator (for visualization demo)
def get_latest_data():
    # In a real scenario, this would read from a shared CSV or database
    new_epoch = st.session_state.epoch + 10
    pde_loss = 0.5 * np.exp(-new_epoch / 500) + 0.05 * np.random.rand()
    bnd_loss = 0.3 * np.exp(-new_epoch / 300) + 0.02 * np.random.rand()
    
    new_row = pd.DataFrame({'epoch': [new_epoch], 'pde_loss': [pde_loss], 'bnd_loss': [bnd_loss]})
    st.session_state.loss_history = pd.concat([st.session_state.loss_history, new_row], ignore_index=True)
    st.session_state.epoch = new_epoch
    
    # Randomly add a mutation
    if np.random.rand() > 0.95:
        st.session_state.mutations.append({
            'time': time.strftime("%H:%M:%S"),
            'edge': f"Layer 0: ({np.random.randint(0,3)}, 0)",
            'expr': "torch.pow(x, 2)",
            'reason': "Detected quadratic volatility smile."
        })

# Update data
get_latest_data()

# Layout: Row 1 - Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("PDE Residual Loss")
    fig_pde = go.Figure()
    fig_pde.add_trace(go.Scatter(x=st.session_state.loss_history['epoch'], 
                                 y=st.session_state.loss_history['pde_loss'],
                                 mode='lines', name='PDE Loss', line=dict(color='firebrick')))
    fig_pde.update_layout(xaxis_title="Epoch", yaxis_title="MSE")
    st.plotly_chart(fig_pde, use_container_width=True)

with col2:
    st.subheader("Boundary Condition Loss")
    fig_bnd = go.Figure()
    fig_bnd.add_trace(go.Scatter(x=st.session_state.loss_history['epoch'], 
                                 y=st.session_state.loss_history['bnd_loss'],
                                 mode='lines', name='Bnd Loss', line=dict(color='royalblue')))
    fig_bnd.update_layout(xaxis_title="Epoch", yaxis_title="MSE")
    st.plotly_chart(fig_bnd, use_container_width=True)

# Layout: Row 2 - Mutation Log & Reasoning
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("🛠️ Topological Mutation Log")
    if st.session_state.mutations:
        for m in reversed(st.session_state.mutations[-5:]):
            st.info(f"**{m['time']}** - Swapped {m['edge']} to `{m['expr']}`\n\n*Reason: {m['reason']}*")
    else:
        st.write("No mutations applied yet.")

with col4:
    st.subheader("🧠 LiuClaw Agent reasoning")
    st.text_area("Live Stream", 
                 value="Analyzing B-spline activation kurtosis...\nDetected structural instability in Edge (0, 0).\nEvaluating symbolic candidates: [exp, pow2, sin]...\nSelecting torch.pow(x, 2) for C2 continuity and alignment with volatility skew.", 
                 height=200)

# Sidebar updates
status_placeholder.success("Active: H200 Training Loop Running")
regime_placeholder.metric("Current Regime", "Low Volatility", delta="-2.1%")

# Auto-refresh logic
time.sleep(1)
st.rerun()
