"""
Pytest configuration and shared fixtures for JobFit AI.
"""
import pytest
import pandas as pd
from typing import List

@pytest.fixture
def sample_resumes() -> List[str]:
    """List of realistic resume texts with skills."""
    return [
        "Data Scientist with 5 years of experience in Python, pandas, and scikit-learn. Familiar with deep learning using PyTorch. Good communication skills.",
        "Frontend Developer with 3 years of experience in React, JavaScript, and HTML/CSS.",
        "Data Engineer specializing in SQL, Python, Airflow, and AWS. Built ETL pipelines and data warehouses.",
        "Machine Learning Engineer experienced in Python, scikit-learn, and model deployment on AWS.",
        "Software Engineer with Java, Spring Boot, and SQL experience. Strong backend skills.",
        "Data Analyst proficient in SQL, Tableau, and Excel. Excellent communication skills.",
        "Cloud Architect with expertise in AWS, Azure, and infrastructure as code.",
        "DevOps Engineer skilled in Docker, Kubernetes, CI/CD, and Python scripting.",
        "Full Stack Developer using Node.js, React, and MongoDB.",
        "Python Developer with experience in Django, REST APIs, and PostgreSQL."
    ]

@pytest.fixture
def sample_job() -> str:
    """A sample job description text."""
    return "We are looking for a Data Scientist experienced in Python, Machine Learning, and SQL. Nice to have: AWS."

@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    """A pandas DataFrame simulating resume-to-job matches."""
    return pd.DataFrame({
        "resume_id": [1, 2, 3],
        "job_id": [101, 101, 101],
        "is_shortlisted": [1, 0, 1]
    })

if __name__ == "__main__":
    pytest.main(["-v", __file__])
