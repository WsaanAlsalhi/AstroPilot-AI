# src/decision_engine.py - Enhanced with detailed analysis
import random
from datetime import datetime

class DecisionEngine:
    def __init__(self):
        """
        Initialize the decision engine with mission parameters and terrain knowledge.
        """
        self.terrain_risk_map = {
            'rocky': 0.8,
            'crater': 0.9,
            'sandy': 0.3,
            'smooth': 0.1,
            'bright dune': 0.5,
            'dark dune': 0.5,
            'impact ejecta': 0.7,
            'slope streak': 0.6,
            'spider': 0.4,
            'swiss cheese': 0.3
        }
        
        self.terrain_descriptions = {
            'rocky': "The terrain consists of large rocks and boulders that may impede wheel traction and increase energy consumption. Navigation requires careful path planning and obstacle avoidance.",
            'crater': "Impact crater with steep slopes and loose debris. High risk of wheel slip and potential entrapment. Recommended to maintain safe distance from crater edges.",
            'sandy': "Fine sand and dust particles. Good traction but potential for wheel sinkage in deeper areas. Energy efficiency is moderate.",
            'smooth': "Flat, even terrain with minimal obstacles. Optimal for high-speed travel and energy-efficient navigation. Low risk of mechanical issues.",
            'bright dune': "Light-colored sand dunes with moderate slopes. Requires careful navigation to avoid sliding. Good visibility for optical sensors.",
            'dark dune': "Dark sand dunes with steeper slopes. Reduced visibility for sensors. Higher risk of navigation errors.",
            'impact ejecta': "Debris field from impact event. Contains sharp fragments that may damage wheels. Requires slow, careful navigation.",
            'slope streak': "Moderate slope with loose surface material. Risk of sliding and reduced traction. Recommended to avoid steep sections.",
            'spider': "Fractured terrain with interconnected cracks. Risk of wheel entrapment. Requires slow, deliberate navigation.",
            'swiss cheese': "Porous, uneven terrain with small pits and voids. Moderate risk of wheel entrapment. Recommended to maintain steady speed."
        }
        
        self.terrain_recommendations = {
            'rocky': {
                'speed': "Reduce speed to 0.5 m/s",
                'route': "Choose path with fewer visible obstacles",
                'energy': "Expect 15-20% higher energy consumption",
                'priority': "Obstacle detection and avoidance"
            },
            'crater': {
                'speed': "Reduce speed to 0.3 m/s",
                'route': "Stay at least 5 meters from crater edge",
                'energy': "Expect 25-30% higher energy consumption",
                'priority': "Slope assessment and traction control"
            },
            'sandy': {
                'speed': "Maintain speed at 0.8 m/s",
                'route': "Follow path with densest sand",
                'energy': "Expect 5-10% higher energy consumption",
                'priority': "Wheel slip monitoring"
            },
            'smooth': {
                'speed': "Increase speed to 1.2 m/s",
                'route': "Direct path recommended",
                'energy': "Optimal energy efficiency",
                'priority': "Sensor calibration for speed"
            },
            'bright dune': {
                'speed': "Reduce speed to 0.6 m/s",
                'route': "Follow dune ridges for better traction",
                'energy': "Expect 10-15% higher energy consumption",
                'priority': "Slope navigation"
            },
            'dark dune': {
                'speed': "Reduce speed to 0.4 m/s",
                'route': "Avoid steep dune faces",
                'energy': "Expect 15-20% higher energy consumption",
                'priority': "Enhanced sensor processing"
            },
            'impact ejecta': {
                'speed': "Reduce speed to 0.3 m/s",
                'route': "Navigate around debris fields",
                'energy': "Expect 30-40% higher energy consumption",
                'priority': "Hazard avoidance"
            },
            'slope streak': {
                'speed': "Reduce speed to 0.4 m/s",
                'route': "Traverse diagonally across slope",
                'energy': "Expect 20-25% higher energy consumption",
                'priority': "Traction management"
            },
            'spider': {
                'speed': "Reduce speed to 0.3 m/s",
                'route': "Navigate through least fractured areas",
                'energy': "Expect 20-30% higher energy consumption",
                'priority': "Wheel slip and entrapment prevention"
            },
            'swiss cheese': {
                'speed': "Maintain speed at 0.7 m/s",
                'route': "Follow path with evenly spaced terrain",
                'energy': "Expect 10-15% higher energy consumption",
                'priority': "Void detection"
            }
        }

    def assess_risk(self, terrain_label, battery_level=80, communication_delay=10, mission_priority='science'):
        """
        Enhanced risk assessment with detailed analysis.
        """
        # Base risk scores
        terrain_risk = self.terrain_risk_map.get(terrain_label, 0.5)
        battery_risk = 1.0 - (battery_level / 100.0)
        comm_risk = min(communication_delay / 30.0, 1.0)
        
        # Priority modifiers
        priority_modifiers = {
            'science': 1.0,
            'survival': 1.3,
            'efficiency': 0.8,
            'exploration': 1.1
        }
        priority_mod = priority_modifiers.get(mission_priority, 1.0)
        
        # Weighted risk
        overall_risk = (terrain_risk * 0.5 + battery_risk * 0.3 + comm_risk * 0.2) * priority_mod
        
        # Determine decision
        if overall_risk < 0.25:
            decision = "CONTINUE_MISSION"
            confidence = "HIGH"
            action = "Proceed with current route at optimal speed"
        elif overall_risk < 0.4:
            decision = "CONTINUE_MISSION"
            confidence = "MEDIUM"
            action = "Proceed with caution, monitor systems closely"
        elif overall_risk < 0.55:
            decision = "CHANGE_ROUTE"
            confidence = "MEDIUM"
            action = "Consider alternative path with lower risk"
        elif overall_risk < 0.7:
            decision = "CAUTION"
            confidence = "LOW"
            action = "Reduce speed significantly and increase sensor monitoring"
        else:
            decision = "SAFE_MODE"
            confidence = "CRITICAL"
            action = "Enter safe mode, stop and wait for instructions"
        
        # Enhanced analysis
        terrain_info = self.terrain_descriptions.get(terrain_label, "Unknown terrain type.")
        recommendations = self.terrain_recommendations.get(terrain_label, {})
        
        # Generate detailed recommendations
        detailed_recommendations = []
        if recommendations:
            detailed_recommendations.append(f"🛞 Speed: {recommendations.get('speed', 'Maintain current speed')}")
            detailed_recommendations.append(f"🗺️ Route: {recommendations.get('route', 'Follow planned route')}")
            detailed_recommendations.append(f"⚡ Energy: {recommendations.get('energy', 'Monitor energy levels')}")
            detailed_recommendations.append(f"🎯 Priority: {recommendations.get('priority', 'Standard operations')}")
        
        # Battery status description
        if battery_level > 70:
            battery_status = "Good - Sufficient power for extended operations"
        elif battery_level > 40:
            battery_status = "Moderate - Power needs to be monitored"
        else:
            battery_status = "Critical - Immediate power conservation required"
        
        # Communication status description
        if communication_delay < 5:
            comm_status = "Excellent - Real-time communication available"
        elif communication_delay < 15:
            comm_status = "Moderate - Some delay in commands"
        else:
            comm_status = "Limited - Significant communication lag"
        
        return {
            'terrain_label': terrain_label,
            'terrain_risk': terrain_risk,
            'battery_risk': battery_risk,
            'comm_risk': comm_risk,
            'overall_risk': overall_risk,
            'decision': decision,
            'confidence': confidence,
            'action': action,
            'terrain_description': terrain_info,
            'recommendations': detailed_recommendations,
            'battery_status': battery_status,
            'comm_status': comm_status,
            'mission_priority': mission_priority
        }
    
    def generate_mission_recommendation(self, risk_assessment):
        """
        Generate a comprehensive mission recommendation.
        """
        decision = risk_assessment['decision']
        action = risk_assessment['action']
        confidence = risk_assessment['confidence']
        terrain = risk_assessment['terrain_label']
        
        base_recommendations = {
            'CONTINUE_MISSION': f"✅ Proceed with mission. Terrain ({terrain}) is manageable. {action}.",
            'CHANGE_ROUTE': f"🔄 Consider changing route. Terrain ({terrain}) has moderate risk. {action}.",
            'CAUTION': f"⚠️ Exercise caution. Terrain ({terrain}) is challenging. {action}.",
            'SAFE_MODE': f"🚨 EMERGENCY: Entering safe mode. Terrain ({terrain}) is hazardous. {action}."
        }
        
        return base_recommendations.get(decision, f"Status unknown. {action}")