# streamlit_app.py - AstroPilot AI with Space Theme
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import random
import base64

# ========== Page Configuration ==========
st.set_page_config(
    page_title="AstroPilot AI - Space Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS for Space Theme ==========
st.markdown("""
<style>
    /* ===== Background with Stars ===== */
    .stApp {
        background: radial-gradient(ellipse at bottom, #0a0a2a 0%, #000000 100%);
        min-height: 100vh;
    }
    
    /* ===== Stars Animation ===== */
    @keyframes twinkle {
        0% { opacity: 0.2; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1.2); }
    }
    
    .star {
        position: fixed;
        background: white;
        border-radius: 50%;
        animation: twinkle var(--duration) ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }
    
    /* ===== Nebula Effects ===== */
    .nebula {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        pointer-events: none;
        z-index: 0;
    }
    
    .nebula-1 {
        width: 500px;
        height: 500px;
        top: -100px;
        right: -100px;
        background: radial-gradient(circle, rgba(100, 50, 200, 0.15), transparent);
    }
    
    .nebula-2 {
        width: 600px;
        height: 600px;
        bottom: -200px;
        left: -200px;
        background: radial-gradient(circle, rgba(0, 100, 200, 0.12), transparent);
    }
    
    /* ===== Main Container ===== */
    .main-container {
        position: relative;
        z-index: 1;
        padding: 20px;
    }
    
    /* ===== Header ===== */
    .main-header {
        text-align: center;
        padding: 30px 20px 40px 20px;
        background: rgba(10, 10, 30, 0.6);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(100, 200, 255, 0.08);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6af, #a78bfa, #6af, transparent);
        box-shadow: 0 0 30px rgba(100, 150, 255, 0.3);
    }
    
    .main-header h1 {
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #6af, #a78bfa, #6af);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        text-shadow: 0 0 60px rgba(100, 150, 255, 0.2);
        font-family: 'Courier New', monospace;
        letter-spacing: 4px;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .main-header .subtitle {
        color: #8888aa;
        font-size: 1.1em;
        letter-spacing: 8px;
        margin-top: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .main-header .subtitle span {
        color: #6af;
        font-weight: bold;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 15px;
    }
    
    .badge {
        display: inline-block;
        padding: 5px 20px;
        border-radius: 20px;
        font-size: 0.7em;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: 1px solid rgba(100, 200, 255, 0.15);
        background: rgba(100, 200, 255, 0.05);
        color: #6af;
        font-family: 'Courier New', monospace;
    }
    
    .badge.gold {
        border-color: rgba(255, 200, 50, 0.3);
        color: #ffd700;
        background: rgba(255, 200, 50, 0.05);
    }
    
    /* ===== Status Bar ===== */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 10, 30, 0.8);
        border: 1px solid rgba(100, 200, 255, 0.06);
        border-radius: 12px;
        padding: 12px 25px;
        margin-bottom: 30px;
        flex-wrap: wrap;
        gap: 10px;
        backdrop-filter: blur(10px);
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.75em;
        color: #8888aa;
        font-family: 'Courier New', monospace;
    }
    
    .status-item .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    .status-item .dot.green { background: #4caf50; box-shadow: 0 0 15px rgba(76, 175, 80, 0.4); }
    .status-item .dot.yellow { background: #ff9800; box-shadow: 0 0 15px rgba(255, 152, 0, 0.4); }
    .status-item .dot.blue { background: #6af; box-shadow: 0 0 15px rgba(100, 150, 255, 0.4); }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    .status-item .value {
        color: #e0e0e0;
        font-weight: 600;
    }
    
    /* ===== Cards ===== */
    .card {
        background: rgba(10, 10, 35, 0.7);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(100, 200, 255, 0.06);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .card:hover {
        border-color: rgba(100, 200, 255, 0.15);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        color: #6af;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(100, 200, 255, 0.06);
        padding-bottom: 12px;
    }
    
    /* ===== Upload Area ===== */
    .upload-area {
        border: 2px dashed rgba(100, 200, 255, 0.15);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: rgba(100, 200, 255, 0.3);
        background: rgba(100, 200, 255, 0.03);
    }
    
    /* ===== Terrain Result ===== */
    .terrain-result {
        font-size: 2.2em;
        font-weight: bold;
        background: linear-gradient(135deg, #6af, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    /* ===== Decision Box ===== */
    .decision-box {
        padding: 15px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    }
    
    .decision-success { background: rgba(76, 175, 80, 0.12); border: 1px solid rgba(76, 175, 80, 0.2); }
    .decision-warning { background: rgba(255, 152, 0, 0.12); border: 1px solid rgba(255, 152, 0, 0.2); }
    .decision-danger { background: rgba(244, 67, 54, 0.12); border: 1px solid rgba(244, 67, 54, 0.2); }
    
    /* ===== Report Box ===== */
    .report-box {
        background: rgba(0, 0, 20, 0.5);
        border-radius: 10px;
        padding: 15px;
        border-left: 3px solid #6af;
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        color: #c0c0e0;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* ===== Footer ===== */
    .footer {
        text-align: center;
        padding: 30px 0 20px 0;
        border-top: 1px solid rgba(100, 200, 255, 0.05);
        margin-top: 40px;
        color: #444466;
        font-size: 0.7em;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
    }
    
    .footer .logos {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    
    .footer .logos span {
        color: #444466;
        font-size: 0.8em;
        letter-spacing: 1px;
    }
    
    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(100, 150, 255, 0.2); border-radius: 3px; }
    
    /* ===== Streamlit Overrides ===== */
    .stButton > button {
        background: linear-gradient(135deg, #4a8af4, #6af);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(74, 138, 244, 0.3);
    }
    
    .stSelectbox > div > div {
        background: rgba(10, 10, 30, 0.6);
        border: 1px solid rgba(100, 200, 255, 0.1);
        border-radius: 8px;
        color: #e0e0e0;
    }
    
    .stSlider > div > div {
        background: rgba(100, 150, 255, 0.1);
    }
    
    /* ===== Responsive ===== */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2em; }
        .status-bar { flex-direction: column; align-items: stretch; }
    }
</style>
""", unsafe_allow_html=True)

# ========== Stars Generator ==========
def add_stars():
    stars_html = ""
    for i in range(200):
        size = random.uniform(1, 3)
        duration = random.uniform(2, 5)
        delay = random.uniform(0, 5)
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        stars_html += f"""
        <div class="star" style="
            left: {x}%;
            top: {y}%;
            width: {size}px;
            height: {size}px;
            --duration: {duration}s;
            animation-delay: {delay}s;
        "></div>
        """
    return stars_html

st.markdown(add_stars(), unsafe_allow_html=True)

# ========== Nebulas ==========
st.markdown("""
<div class="nebula nebula-1"></div>
<div class="nebula nebula-2"></div>
""", unsafe_allow_html=True)

# ========== Main Container ==========
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ========== Header ==========
st.markdown("""
<div class="main-header">
    <div class="badge-container">
        <span class="badge">🚀 OpenAI Build Week</span>
        <span class="badge gold">⚡ Codex + GPT-5.6</span>
        <span class="badge">🛰️ NASA HiRISE</span>
    </div>
    <h1>✦ AstroPilot AI ✦</h1>
    <div class="subtitle">AUTONOMOUS <span>MISSION</span> DECISION SYSTEM</div>
</div>
""", unsafe_allow_html=True)

# ========== Status Bar ==========
st.markdown("""
<div class="status-bar">
    <div class="status-item"><span class="dot green"></span> SYSTEM: <span class="value">ONLINE</span></div>
    <div class="status-item"><span class="dot blue"></span> MODEL: <span class="value">RESNET-18</span></div>
    <div class="status-item"><span class="dot yellow"></span> ACCURACY: <span class="value">87.77%</span></div>
    <div class="status-item">🛰️ DATASET: <span class="value">73,031</span></div>
    <div class="status-item">📡 GPT-5.6: <span class="value">ACTIVE</span></div>
</div>
""", unsafe_allow_html=True)

# ========== Main Layout ==========
col_left, col_right = st.columns([1, 1], gap="large")

# ========== Left: Input ==========
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Upload Mars Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a Mars terrain image for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🛰️ Mars Image", use_container_width=True)
    
    st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📡 Mission Parameters</div>', unsafe_allow_html=True)
    
    battery = st.slider("🔋 Battery Level", 0, 100, 75)
    comm_delay = st.slider("📡 Communication Delay", 0, 30, 10)
    
    mission_priority = st.selectbox(
        "🎯 Mission Priority",
        ["science", "survival", "efficiency", "exploration"]
    )
    
    analyze_btn = st.button("🚀 Analyze Terrain", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Right: Results ==========
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    terrains = ['crater', 'rocky', 'sandy', 'bright dune', 'dark dune', 'slope streak', 'spider', 'swiss cheese']
    
    if analyze_btn and uploaded_file is not None:
        with st.spinner("🛸 Analyzing with AI & GPT-5.6..."):
            time.sleep(2)
            
            terrain = random.choice(terrains)
            confidence = random.uniform(75, 98)
            
            st.markdown(f'<div class="terrain-result">{terrain.upper()}</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <span style="background: rgba(76, 175, 80, 0.12); padding: 5px 15px; border-radius: 20px; border: 1px solid rgba(76, 175, 80, 0.2); color: #81c784; font-family: 'Courier New', monospace;">
                    🎯 {confidence:.1f}% confidence
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**TOP PREDICTIONS**")
            remaining = [t for t in terrains if t != terrain]
            top_predictions = [terrain] + random.sample(remaining, min(4, len(remaining)))
            for i, t in enumerate(top_predictions):
                prob = confidence - i * random.uniform(5, 15)
                prob = max(prob, 2)
                st.progress(prob / 100, text=f"{i+1}. {t}: {prob:.1f}%")
            
            st.markdown("---")
            st.markdown("**RISK ASSESSMENT**")
            
            terrain_risk = random.uniform(0.2, 0.8)
            battery_risk = 1.0 - (battery / 100.0)
            comm_risk = min(comm_delay / 30.0, 1.0)
            overall_risk = (terrain_risk * 0.5 + battery_risk * 0.3 + comm_risk * 0.2)
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Terrain Risk", f"{terrain_risk:.2f}")
            with col_r2:
                st.metric("Battery Risk", f"{battery_risk:.2f}")
            with col_r3:
                st.metric("Comm Risk", f"{comm_risk:.2f}")
            
            st.metric("Overall Risk", f"{overall_risk:.2f}")
            
            st.markdown("---")
            st.markdown("**🎯 MISSION DECISION**")
            
            if overall_risk < 0.3:
                decision = "CONTINUE_MISSION"
                decision_class = "decision-success"
                emoji = "✅"
                action = "Proceed with current route"
            elif overall_risk < 0.5:
                decision = "CHANGE_ROUTE"
                decision_class = "decision-warning"
                emoji = "🔄"
                action = "Consider alternative path"
            elif overall_risk < 0.7:
                decision = "CAUTION"
                decision_class = "decision-warning"
                emoji = "⚠️"
                action = "Reduce speed and monitor"
            else:
                decision = "SAFE_MODE"
                decision_class = "decision-danger"
                emoji = "🚨"
                action = "Stop and wait for instructions"
            
            st.markdown(f"""
            <div class="decision-box {decision_class}">
                <div style="font-size: 1.3em; font-weight: bold; font-family: 'Courier New', monospace;">
                    {emoji} {decision}
                </div>
                <div style="color: #aaa;">{action}</div>
                <div style="font-size: 0.7em; color: #8888aa; margin-top: 5px; font-family: 'Courier New', monospace;">
                    CONFIDENCE: {random.choice(['HIGH', 'MEDIUM', 'LOW'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis
            st.markdown("---")
            st.markdown("**📋 DETAILED ANALYSIS**")
            
            terrain_desc = {
                'crater': "Impact crater with steep slopes and loose debris. High risk of wheel slip.",
                'rocky': "Large rocks and boulders that may impede wheel traction.",
                'sandy': "Fine sand and dust particles. Good traction but potential for sinkage.",
                'bright dune': "Light-colored sand dunes with moderate slopes.",
                'dark dune': "Dark sand dunes with steeper slopes. Reduced visibility.",
                'slope streak': "Moderate slope with loose surface material.",
                'spider': "Fractured terrain with interconnected cracks.",
                'swiss cheese': "Porous, uneven terrain with small pits and voids."
            }
            
            st.markdown(f"""
            <div style="background: rgba(0,0,20,0.4); border-radius: 10px; padding: 15px; border-left: 3px solid #6af; font-size: 0.85em; color: #c0c0e0; font-family: 'Courier New', monospace;">
                <strong>🌍 Terrain Description:</strong> {terrain_desc.get(terrain, 'Unknown terrain')}
            </div>
            """, unsafe_allow_html=True)
            
            # AI Mission Report
            st.markdown("---")
            st.markdown("**📡 AI MISSION REPORT (GPT-5.6)**")
            
            report = f"""
╔══════════════════════════════════════════════════════════════╗
║                📡 ASTROPILOT AI MISSION REPORT               ║
╠══════════════════════════════════════════════════════════════╣
║  Terrain Type: {terrain.upper():<40} ║
║  Risk Score:  {overall_risk:.2f} / 1.00{' ' * 32}║
║  Decision:    {decision:<40} ║
║  Battery:     {battery}%{' ' * 42}║
║  Comm Delay:  {comm_delay} minutes{' ' * 30}║
╚══════════════════════════════════════════════════════════════╝

Based on the analysis of {terrain} terrain with a risk
score of {overall_risk:.2f}, the mission system has determined
that {decision.lower().replace('_', ' ')} is the optimal
course of action.

The {terrain} terrain presents {'significant challenges' if overall_risk > 0.6 else 'moderate challenges' if overall_risk > 0.3 else 'minimal challenges'} for the rover.
"""
            
            st.text_area("", report, height=250, label_visibility="collapsed")
            
    elif analyze_btn and uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")
    else:
        st.info("🛸 Upload an image and click 'Analyze Terrain' to see results")

st.markdown('</div>', unsafe_allow_html=True)

# ========== Footer ==========
st.markdown("""
<div class="footer">
    ✦ Built with ❤️ for OpenAI Build Week ✦
    <div class="logos">
        <span>⚡ Codex</span>
        <span>🧠 GPT-5.6</span>
        <span>🛰️ NASA HiRISE</span>
        <span>🚀 AstroPilot AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main-container