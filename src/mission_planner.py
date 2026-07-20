# src/mission_planner.py
import random
from datetime import datetime, timedelta

class MissionPlanner:
    def __init__(self):
        """
        Initialize mission planner with sample mission scenarios.
        """
        self.mission_types = ['Mars Rover Exploration', 'Satellite Observation', 'Lander Operations']
        self.terrain_conditions = ['rocky', 'sandy', 'crater', 'smooth', 'bright dune', 'dark dune']
        self.weather_conditions = ['clear', 'dust storm', 'cloudy', 'extreme cold']
        
    def generate_mission_scenario(self, mission_type=None, terrain=None, battery=None, comm_delay=None):
        """
        Generate a realistic mission scenario.
        """
        if mission_type is None:
            mission_type = random.choice(self.mission_types)
        
        if terrain is None:
            terrain = random.choice(self.terrain_conditions)
        
        if battery is None:
            battery = random.randint(40, 95)
        
        if comm_delay is None:
            comm_delay = random.randint(5, 25)
        
        weather = random.choice(self.weather_conditions)
        
        scenario = {
            'mission_type': mission_type,
            'terrain': terrain,
            'battery_percentage': battery,
            'communication_delay_minutes': comm_delay,
            'weather': weather,
            'timestamp': datetime.now().isoformat(),
            'mission_duration_hours': random.randint(2, 24)
        }
        
        return scenario
    
    def create_mission_plan(self, scenario):
        """
        Create a detailed mission plan based on the scenario.
        """
        plan = f"""
🚀 MISSION PLAN: {scenario['mission_type']}
──────────────────────────────────────────────────

📅 Timestamp: {scenario['timestamp']}
📍 Current Terrain: {scenario['terrain']}
🔋 Battery Level: {scenario['battery_percentage']}%
📡 Comm Delay: {scenario['communication_delay_minutes']} minutes
☁️ Weather: {scenario['weather']}
⏱️ Duration: {scenario['mission_duration_hours']} hours

📋 Objectives:
  1. Navigate through {scenario['terrain']} terrain
  2. Collect surface samples and images
  3. Transmit data back to mission control
  4. Maintain safe operations

📊 Priority Tasks:
  - Primary: Safe navigation
  - Secondary: Data collection
  - Tertiary: Communication
"""
        return plan
    
    def simulate_mission_scenario(self):
        """
        Generate and print a complete mission scenario.
        """
        scenario = self.generate_mission_scenario()
        plan = self.create_mission_plan(scenario)
        print(plan)
        return scenario, plan