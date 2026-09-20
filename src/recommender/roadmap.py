# Skill database with estimated hours and resources
SKILL_DB = {
    "python": {
        "hours": 40,
        "free_resource": "FreeCodeCamp - Scientific Computing with Python",
        "paid_resource": "Coursera - Python for Everybody Specialization (University of Michigan)"
    },
    "sql": {
        "hours": 20,
        "free_resource": "SQLZoo & Mode Analytics SQL Tutorial",
        "paid_resource": "DataCamp - Data Analyst with SQL Track"
    },
    "machine learning": {
        "hours": 60,
        "free_resource": "Google Machine Learning Crash Course",
        "paid_resource": "Coursera - Machine Learning Specialization by Andrew Ng (DeepLearning.AI)"
    },
    "pandas": {
        "hours": 15,
        "free_resource": "Kaggle Pandas Micro-course",
        "paid_resource": "Udemy - Data Analysis with Pandas and Python"
    },
    "aws": {
        "hours": 50,
        "free_resource": "AWS Skill Builder (Cloud Practitioner Pathway)",
        "paid_resource": "A Cloud Guru / Pluralsight - AWS Certified Solutions Architect"
    },
    "docker": {
        "hours": 15,
        "free_resource": "Docker Official Quickstart Guide & YouTube (TechWorld with Nana)",
        "paid_resource": "Udemy - Docker Mastery: with Kubernetes +Swarm"
    },
    "kubernetes": {
        "hours": 40,
        "free_resource": "Kubernetes.io Interactive Tutorials",
        "paid_resource": "Udemy - Certified Kubernetes Administrator (CKA)"
    },
    "react": {
        "hours": 45,
        "free_resource": "React Official Docs & Scrimba Free React Course",
        "paid_resource": "Frontend Masters - Complete React Track"
    },
    "pytorch": {
        "hours": 35,
        "free_resource": "PyTorch Official Tutorials & Fast.ai",
        "paid_resource": "Udacity - Deep Learning Nanodegree"
    },
    "tensorflow": {
        "hours": 35,
        "free_resource": "TensorFlow Official Tutorials",
        "paid_resource": "Coursera - DeepLearning.AI TensorFlow Developer"
    },
    "java": {
        "hours": 50,
        "free_resource": "Codecademy Learn Java (Free Tier)",
        "paid_resource": "Udemy - Java Programming Masterclass"
    }
}

def generate_roadmap(skills, hours_per_day, budget):
    """
    Generates a personalized study roadmap for a list of skills.
    
    Args:
        skills (list of str): Missing skills to learn.
        hours_per_day (int): User's available study time.
        budget (str): "Free" or "Paid"
        
    Returns:
        list of dict: Step-by-step roadmap with days and resources.
    """
    roadmap = []
    current_day = 1
    
    for skill in skills:
        skill_lower = skill.lower().strip()
        
        # Look up skill or use defaults
        if skill_lower in SKILL_DB:
            total_hours = SKILL_DB[skill_lower]["hours"]
            resource = SKILL_DB[skill_lower]["free_resource"] if budget == "Free" else SKILL_DB[skill_lower]["paid_resource"]
        else:
            total_hours = 20  # Default 20 hours
            resource = f"YouTube Crash Course / Official Docs for {skill}" if budget == "Free" else f"Udemy / Coursera Highly Rated Course for {skill}"
            
        days_needed = max(1, round(total_hours / hours_per_day))
        
        roadmap.append({
            "skill": skill,
            "total_hours": total_hours,
            "days_needed": days_needed,
            "start_day": current_day,
            "end_day": current_day + days_needed - 1,
            "resource": resource
        })
        
        current_day += days_needed
        
    return roadmap
