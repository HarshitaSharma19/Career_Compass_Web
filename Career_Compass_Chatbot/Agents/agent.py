import os
from utils.common import (
    situation_question_generation_prompt,
    generate_storytelling_questions,
    evaluate_depression_level,
    generate_career_recommendations,
    generate_learning_roadmap,
    analyze_career_consultation,
    analyze_skills_gap
)
from dotenv import load_dotenv
import google.generativeai as gen_ai



load_dotenv()

class Agents:
    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        gen_ai.configure(api_key=self.GOOGLE_API_KEY)
        self.model = gen_ai.GenerativeModel('gemini-pro')

    
    def situation_question_generation_agent(self,data):
        prompt = situation_question_generation_prompt(data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_storytelling_questions_agent(self,data):
        prompt = generate_storytelling_questions(data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def evaluate_user(self,data):
        prompt = evaluate_depression_level(data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_career_recommendations_agent(self, user_data, preferences):
        """Generate career recommendations based on user profile and preferences"""
        prompt = generate_career_recommendations(user_data, preferences)
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_learning_roadmap_agent(self, career_path, user_data):
        """Generate a personalized learning roadmap for the selected career path"""
        prompt = generate_learning_roadmap(career_path, user_data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def career_consultation_agent(self, question, user_data):
        """Process user questions during live AI consultation"""
        prompt = analyze_career_consultation(question, user_data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def skills_gap_analysis_agent(self, career_path, user_data):
        """Analyze the gap between user's current skills and required skills"""
        prompt = analyze_skills_gap(career_path, user_data)
        response = self.model.generate_content(prompt)
        return response.text
    
    

