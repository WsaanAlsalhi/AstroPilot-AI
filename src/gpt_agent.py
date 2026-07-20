# src/gpt_agent.py - Enhanced with detailed analysis
import json
from datetime import datetime

class GPTAgent:
    def __init__(self, api_key=None):
        """
        Initialize the GPT Agent.
        """
        self.api_key = api_key
        self.use_real_api = False if api_key is None else True
    
    def generate_mission_report(self, risk_assessment, battery=80, comm_delay=10):
        """
        Generate a comprehensive mission report with detailed analysis.
        """
        decision = risk_assessment['decision']
        terrain = risk_assessment['terrain_label']
        terrain_risk = risk_assessment['terrain_risk']
        overall_risk = risk_assessment['overall_risk']
        terrain_desc = risk_assessment.get('terrain_description', '')
        recommendations = risk_assessment.get('recommendations', [])
        battery_status = risk_assessment.get('battery_status', '')
        comm_status = risk_assessment.get('comm_status', '')
        
        # Build report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        header = f"""
╔══════════════════════════════════════════════════════════════╗
║                📡 ASTROPILOT AI MISSION REPORT               ║
╠══════════════════════════════════════════════════════════════╣
║  Generated: {timestamp}                                      ║
║  Mission Status: {'🟢 NOMINAL' if decision == 'CONTINUE_MISSION' else '🟡 MODERATE RISK' if decision == 'CHANGE_ROUTE' else '🔴 HIGH RISK' if decision == 'CAUTION' else '🚨 CRITICAL'}  ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        # Terrain Analysis
        terrain_section = f"""
┌─────────────────────────────────────────────────────────────┐
│                     🛰️ TERRAIN ANALYSIS                    │
├─────────────────────────────────────────────────────────────┤
│  Terrain Type: {terrain.upper()}                                    │
│  Risk Score:  {terrain_risk:.2f} / 1.00                            │
│                                                             │
│  Description:                                               │
│  {terrain_desc[:80]}...
│                                                             │
│  Classification Confidence: {risk_assessment.get('confidence', 'UNKNOWN')}  │
└─────────────────────────────────────────────────────────────┘
"""
        
        # Mission Parameters
        params_section = f"""
┌─────────────────────────────────────────────────────────────┐
│                     📡 MISSION PARAMETERS                   │
├─────────────────────────────────────────────────────────────┤
│  Battery Level:    {battery}% - {battery_status}              │
│  Comm Delay:       {comm_delay} minutes - {comm_status}      │
│  Mission Priority: {risk_assessment.get('mission_priority', 'science')}  │
│  Overall Risk:     {overall_risk:.2f} / 1.00                        │
└─────────────────────────────────────────────────────────────┘
"""
        
        # Recommendations
        rec_section = f"""
┌─────────────────────────────────────────────────────────────┐
│                     🎯 RECOMMENDATIONS                      │
├─────────────────────────────────────────────────────────────┤
│  DECISION: {decision}                                     │
│  CONFIDENCE: {risk_assessment.get('confidence', 'UNKNOWN')}                │
│  ACTION: {risk_assessment.get('action', 'N/A')}             │
│                                                             │
│  DETAILED RECOMMENDATIONS:                                 │
"""
        for rec in recommendations:
            rec_section += f"  │  {rec}\n"
        
        rec_section += """
└─────────────────────────────────────────────────────────────┘
"""
        
        # Detailed Narrative
        narrative = f"""
┌─────────────────────────────────────────────────────────────┐
│                  📋 DETAILED NARRATIVE                      │
├─────────────────────────────────────────────────────────────┤
│  Based on the analysis of {terrain} terrain with a risk     │
│  score of {terrain_risk:.2f}, the mission system has determined  │
│  that {decision.lower().replace('_', ' ')} is the optimal   │
│  course of action.                                         │
│                                                             │
│  The {terrain} terrain presents {'significant challenges' if terrain_risk > 0.6 else 'moderate challenges' if terrain_risk > 0.3 else 'minimal challenges'} for the   │
│  rover. The {battery}% battery level indicates {battery_status.lower()},  │
│  while the {comm_delay}-minute communication delay suggests   │
│  {comm_status.lower()}.                                     │
│                                                             │
│  Risk assessment:                                          │
│  • Terrain risk: {terrain_risk:.2f}                                  │
│  • Battery risk: {risk_assessment['battery_risk']:.2f}               │
│  • Communication risk: {risk_assessment['comm_risk']:.2f}             │
│                                                             │
│  {risk_assessment.get('action', 'Proceed with caution')}.  │
│                                                             │
│  Mission priority is set to {risk_assessment.get('mission_priority', 'science')}  │
│  which {'increases' if risk_assessment.get('mission_priority') == 'survival' else 'maintains'} the risk sensitivity.  │
└─────────────────────────────────────────────────────────────┘
"""
        
        return header + terrain_section + params_section + rec_section + narrative