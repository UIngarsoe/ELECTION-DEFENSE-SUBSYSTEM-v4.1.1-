import streamlit as st
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any

# ======================================================================
# 1. CORE LOGIC (Based on SSISM Philosophy)
# ======================================================================

class ElectionDefenseEngine:
    """
    ELECTION-DEFENSE-SUBSYSTEM-v4.1.1 Core
    Focuses on Z-Score calculation and SĪLA-Coded Remediation.
    """
    
    def __init__(self, threshold_phi=0.2):
        self.log = []
        self.threshold_phi = threshold_phi # Critical Digital Trust Score threshold
        self.anomalies = []

    def calculate_z_score(self, factors: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculates the weighted Total Risk Score (Z) for an election event.
        Factors: Authority, Urgency, Linguistics, Link/File (R), Time Anomaly (ΔT), etc.
        """
        if not factors or not weights:
            return 0.0

        z_numerator = sum(factors.get(k, 0.0) * weights.get(k, 0.0) for k in weights)
        z_denominator = sum(weights.values())
        
        # Avoid division by zero
        if z_denominator == 0:
            return 0.0
            
        return z_numerator / z_denominator

    def z_to_phi(self, z_score: float) -> float:
        """
        Transforms Z-Score into Digital Trust Score (Phi) using the Sigmoid (Logistic Regression).
        Φ = 1 / (1 + e^(-Z))
        """
        # We adjust the input to the sigmoid to center Z=0 at Phi=0.5.
        # However, for risk, higher Z (risk) should mean lower Phi (trust).
        # We use Φ = 1 - (1 / (1 + e^(-Z))) or adjust Z input.
        # Let's use the standard SSISM form (Sigmoid) where higher Z means higher risk score, 
        # and then invert for the Trust Score (Φ).
        
        # Risk Score (R): Higher Z -> Higher R (closer to 1)
        risk_score = 1 / (1 + np.exp(-z_score))
        
        # Digital Trust Score (Φ): Invert the Risk Score
        phi_score = 1.0 - risk_score
        
        return phi_score

    def issue_defense_protocol(self, z_score: float, phi_score: float, event_name: str, recommendation: str):
        """Records an event and issues a protocol if Phi falls below the threshold."""
        
        protocol_issued = False
        action = "MONITOR"

        if phi_score < self.threshold_phi:
            protocol_issued = True
            action = f"MANDATORY LOCKOUT: {recommendation}"
            
            anomaly_record = {
                "id": len(self.anomalies) + 1,
                "event": event_name,
                "Z_Score": f"{z_score:.4f}",
                "Phi_Score": f"{phi_score:.4f}",
                "Action": action,
                "Timestamp": datetime.now().isoformat()
            }
            self.anomalies.append(anomaly_record)
        
        self.log.append({
            "event": event_name,
            "Z_Score": z_score,
            "Phi_Score": phi_score,
            "Action": action,
            "Protocol_Issued": protocol_issued,
            "Timestamp": datetime.now().isoformat()
        })
        
        return action, protocol_issued

# ======================================================================
# 2. STREAMLIT INTERFACE
# ======================================================================

# Initialize Engine in session state
if 'eds_engine' not in st.session_state:
    st.session_state.eds_engine = ElectionDefenseEngine()

engine = st.session_state.eds_engine

st.set_page_config(page_title="ELECTION DEFENSE SUBSYSTEM v4.1.1", layout="wide")

st.markdown("## 🛡️ ELECTION-DEFENSE-SUBSYSTEM | v4.1.1")
st.caption('***Anomalous Z-Score Detection & SĪLA-Coded Remediation***')
st.markdown("---")

# --- Default Weights ---
DEFAULT_WEIGHTS = {
    "Authority (A)": 0.4,
    "Urgency (U)": 0.3,
    "Linguistics (L)": 0.1,
    "Link/File (R)": 0.1,
    "Time Anomaly (ΔT)": 0.1
}

# --- Sidebar for Engine Configuration ---
st.sidebar.title("Engine Config")
new_threshold = st.sidebar.slider(
    "Critical $\Phi$ Threshold",
    0.01, 0.5, engine.threshold_phi, 0.01,
    help="If Digital Trust Score ($\Phi$) drops below this, a MANDATORY LOCKOUT Protocol is issued."
)
engine.threshold_phi = new_threshold
st.sidebar.info(f"Current Critical $\Phi$: **{engine.threshold_phi:.2f}**")

# --- Main Interface Tabs ---
tab1, tab2 = st.tabs(["📊 Z-Score Analysis", "🚨 Anomaly Log"])

# ----------------------------------
# Tab 1: Z-Score Analysis
# ----------------------------------
with tab1:
    st.header("1. Election Event Risk Assessment")
    st.markdown("Input the weighted factors for a potential threat event (e.g., disinformation campaign, vote manipulation claim) to calculate the Total Risk Score ($Z$) and Digital Trust Score ($\Phi$).")
    
    col_event, col_prot = st.columns([2, 1])
    with col_event:
        event_name = st.text_input("Election Event/Threat Name", "Voter Registration Disinformation Spike")
    with col_prot:
        remediation_protocol = st.selectbox(
            "MANDATORY LOCKOUT Recommendation",
            ["Issue Public Integrity Alert", "Initiate Blockchain Audit", "24-Hour Info Blackout"]
        )

    st.subheader("Weighted Risk Factors (Scale 0.0 to 10.0)")
    
    # Dynamic Factor Input based on Weights keys
    factors = {}
    
    # Use columns to lay out the inputs nicely
    cols = st.columns(len(DEFAULT_WEIGHTS))
    for i, (factor, default_w) in enumerate(DEFAULT_WEIGHTS.items()):
        with cols[i]:
            # Weight is displayed for context but the factor value is user input
            st.markdown(f"**{factor}** (W: {default_w:.1f})")
            factors[factor] = st.slider(factor, 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed", key=f"factor_{i}")
    
    st.markdown("---")

    if st.button("Calculate $\mathbf{Z}$ and Issue Protocol", use_container_width=True, type="primary"):
        weights = DEFAULT_WEIGHTS
        
        # Calculate Z-Score
        z_score = engine.calculate_z_score(factors, weights)
        
        # Calculate Phi-Score
        phi_score = engine.z_to_phi(z_score)

        # Issue Protocol
        action, protocol_issued = engine.issue_defense_protocol(z_score, phi_score, event_name, remediation_protocol)

        st.success("Analysis Complete. Protocol Action Executed.")

        col_z, col_phi = st.columns(2)
        
        with col_z:
            st.metric(
                label="Total Risk Score ($Z$)", 
                value=f"{z_score:.4f}", 
                delta_color="off"
            )

        with col_phi:
            # Color the metric based on safety threshold
            delta_val = f"{(phi_score - engine.threshold_phi):.4f}"
            delta_color = "inverse" if phi_score < engine.threshold_phi else "normal"
            st.metric(
                label="Digital Trust Score ($\Phi = 1 - R$)", 
                value=f"{phi_score:.4f}", 
                delta=f"Diff from $\Phi_{{crit}}$: {delta_val}", 
                delta_color=delta_color
            )
            
        st.markdown("#### **Protocol Action:**")
        if protocol_issued:
            st.error(f"🚨 {action}")
            st.markdown(f"**Reason:** $\Phi$ ({phi_score:.4f}) is below the critical threshold ({engine.threshold_phi:.2f}).")
        else:
            st.info(f"✅ {action} - $\Phi$ is within acceptable limits.")

# ----------------------------------
# Tab 2: Anomaly Log
# ----------------------------------
with tab2:
    st.header("2. SĪLA-Coded Anomaly Lockout Log")
    st.markdown("Records of all events where the Digital Trust Score ($\Phi$) fell below the critical threshold, triggering a **MANDATORY LOCKOUT** protocol.")
    
    if engine.anomalies:
        # Use pandas DataFrame for better visualization
        import pandas as pd
        anomaly_df = pd.DataFrame(engine.anomalies)
        
        # Rename columns for display
        anomaly_df.rename(columns={
            'event': 'Event Name', 
            'Z_Score': 'Z-Score', 
            'Phi_Score': 'Trust ($\Phi$)', 
            'Action': 'Protocol Executed', 
            'Timestamp': 'Time (UTC)'
        }, inplace=True)
        
        st.dataframe(anomaly_df.drop(columns=['id']), use_container_width=True, hide_index=True)
    else:
        st.info("No critical anomalies have been detected yet.")

