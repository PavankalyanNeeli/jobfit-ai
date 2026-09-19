"""
Data loading and synthetic generation module for JobFit AI.

Provides a deterministic synthetic data generator and functions to load/save data.
"""

import random
from typing import Tuple

import pandas as pd

from src.config import (
    N_SYNTHETIC_JOBS,
    N_SYNTHETIC_RESUMES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET_ROLES,
)
from src.features.skills import SKILL_CATALOG, get_skills_for_role


def _generate_fake_name(rng: random.Random) -> str:
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Skyler", "Avery", "Parker"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return f"{rng.choice(first_names)} {rng.choice(last_names)}"


def generate_synthetic_resumes(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic resumes.

    Args:
        n (int): Number of resumes to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing generated resumes.
    """
    rng = random.Random(seed)
    all_skills = list(SKILL_CATALOG.keys())
    
    data = []
    for i in range(1, n + 1):
        target_role = rng.choice(TARGET_ROLES)
        role_skills = list(get_skills_for_role(target_role))
        if not role_skills:
            role_skills = all_skills
        
        # Pick 4-12 skills mostly from the role, some random
        num_skills = rng.randint(4, 12)
        num_role_skills = min(len(role_skills), int(num_skills * 0.8))
        num_random_skills = num_skills - num_role_skills
        
        skills = list(set(rng.sample(role_skills, num_role_skills) + rng.sample(all_skills, num_random_skills)))
        
        experience_years = round(rng.uniform(0.5, 15.0), 1)
        education_level = rng.choice(["High School", "Bachelor", "Master", "PhD"])
        
        # Create a realistic summary
        verbs = ["Developed", "Engineered", "Analyzed", "Led", "Designed", "Implemented", "Built", "Optimized", "Deployed", "Architected"]
        projects = ["machine learning models", "data pipelines", "web applications", "backend systems", "analytics dashboards", "ETL workflows", "recommendation engines", "REST APIs"]
        domains = ["e-commerce", "fintech", "healthcare", "SaaS", "ad-tech", "logistics", "social media"]
        
        # Build rich resume text with many skill mentions for TF-IDF signal
        skill_list_str = ", ".join(skills)
        primary_skills = ", ".join(skills[:3])
        secondary_skills = ", ".join(skills[3:6]) if len(skills) > 3 else ""
        
        sentences = [
            f"{rng.choice(verbs)} {rng.choice(projects)} using {primary_skills} in the {rng.choice(domains)} domain.",
            f"Proficient in {skill_list_str}.",
            f"Strong background with {experience_years} years of industry experience as a {target_role}.",
        ]
        if secondary_skills:
            sentences.append(f"Also experienced with {secondary_skills} for production-grade systems.")
        sentences.append(f"Education: {education_level} degree. Seeking {target_role} roles.")
        if len(skills) > 6:
            sentences.append(f"{rng.choice(verbs)} solutions leveraging {', '.join(skills[6:])}.")
        
        raw_text = " ".join(sentences)
        
        data.append({
            "resume_id": i,
            "name": _generate_fake_name(rng),
            "title": f"Experienced {target_role.title()}",
            "raw_text": raw_text,
            "skills": skills,
            "experience_years": experience_years,
            "education_level": education_level,
            "target_role": target_role
        })
        
    return pd.DataFrame(data)


def generate_synthetic_jobs(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic job postings.

    Args:
        n (int): Number of jobs to generate.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: DataFrame containing generated job postings.
    """
    rng = random.Random(seed)
    
    companies = ["TechCorp", "DataSystems", "InnoSoft", "CloudNet", "AI Dynamics", "Quantum Solutions"]
    seniorities = ["Junior", "Mid-level", "Senior", "Lead", "Principal"]
    
    data = []
    for i in range(1, n + 1):
        role = rng.choice(TARGET_ROLES)
        seniority = rng.choice(seniorities)
        title = f"{seniority} {role.title()}"
        
        role_skills = list(get_skills_for_role(role))
        if not role_skills:
            role_skills = list(SKILL_CATALOG.keys())
            
        num_req = rng.randint(5, min(10, len(role_skills)))
        required_skills = rng.sample(role_skills, num_req)
        
        remaining = list(set(role_skills) - set(required_skills))
        
        num_pref = 0
        preferred_skills = []
        if remaining:
            upper_bound = min(5, len(remaining))
            lower_bound = min(2, upper_bound)
            num_pref = rng.randint(lower_bound, upper_bound)
            preferred_skills = rng.sample(remaining, num_pref)
            
        min_experience = round(rng.uniform(1.0, 8.0), 1)
        company = rng.choice(companies)
        
        description = (
            f"We are {company}, looking for a {title}. "
            f"You will need strong skills in {', '.join(required_skills[:3])}. "
            f"Minimum {min_experience} years of experience required. "
            f"Nice to have: {', '.join(preferred_skills)}. "
            f"Join our fast-growing team and build the future!"
        )
        
        data.append({
            "job_id": i,
            "title": title,
            "description": description,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "min_experience": min_experience,
            "company": company,
            "source": "synthetic"
        })
        
    return pd.DataFrame(data)


def generate_matches(resumes_df: pd.DataFrame, jobs_df: pd.DataFrame, matches_per_resume: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic match labels for resume-job pairs.

    Args:
        resumes_df (pd.DataFrame): The resumes DataFrame.
        jobs_df (pd.DataFrame): The jobs DataFrame.
        matches_per_resume (int): How many jobs to pair each resume with.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: DataFrame containing matches and labels.
    """
    rng = random.Random(seed)
    
    matches = []
    
    job_ids = jobs_df["job_id"].tolist()
    
    for _, resume in resumes_df.iterrows():
        res_id = resume["resume_id"]
        res_skills = set(resume["skills"])
        res_exp = resume["experience_years"]
        
        # Sample jobs for this resume
        sampled_job_ids = rng.sample(job_ids, min(matches_per_resume, len(job_ids)))
        
        for jid in sampled_job_ids:
            job = jobs_df[jobs_df["job_id"] == jid].iloc[0]
            req_skills = set(job["required_skills"])
            job_exp = job["min_experience"]
            
            coverage = len(res_skills & req_skills) / len(req_skills) if req_skills else 0.0
            exp_match = res_exp >= job_exp
            
            # Deterministic randomness
            rand_val = hash(res_id * 1000 + jid) % 100
            
            if coverage > 0.6 and exp_match:
                prob = 80
            elif coverage > 0.4:
                prob = 30
            else:
                prob = 5
                
            is_shortlisted = 1 if rand_val < prob else 0
            
            matches.append({
                "resume_id": res_id,
                "job_id": jid,
                "skill_coverage": round(coverage, 2),
                "experience_match": exp_match,
                "is_shortlisted": is_shortlisted
            })
            
    return pd.DataFrame(matches)


def load_data(source: str = "synthetic") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main entry point to load data.

    Args:
        source (str): Source of data ('synthetic' only currently).

    Returns:
        Tuple of (resumes_df, jobs_df, matches_df)
    """
    if source == "synthetic":
        resumes = generate_synthetic_resumes(n=N_SYNTHETIC_RESUMES, seed=RANDOM_SEED)
        jobs = generate_synthetic_jobs(n=N_SYNTHETIC_JOBS, seed=RANDOM_SEED)
        matches = generate_matches(resumes, jobs, seed=RANDOM_SEED)
        return resumes, jobs, matches
    else:
        raise ValueError(f"Unknown data source: {source}")


def save_processed_data(resumes: pd.DataFrame, jobs: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Save the processed DataFrames to disk.

    Args:
        resumes (pd.DataFrame): Resumes DataFrame.
        jobs (pd.DataFrame): Jobs DataFrame.
        matches (pd.DataFrame): Matches DataFrame.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    resumes.to_csv(PROCESSED_DIR / "resumes.csv", index=False)
    jobs.to_csv(PROCESSED_DIR / "jobs.csv", index=False)
    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    print(f"Data saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    print("Generating synthetic data...")
    res_df, jobs_df, match_df = load_data()
    
    print(f"\nGenerated {len(res_df)} resumes.")
    print(f"Generated {len(jobs_df)} jobs.")
    print(f"Generated {len(match_df)} matches.")
    
    print(f"\nShortlisted matches: {match_df['is_shortlisted'].sum()} ({match_df['is_shortlisted'].mean():.1%})")
    print("\nSample Resumes:")
    print(res_df.head(2))
    
    print("\nSample Matches:")
    print(match_df.head(5))
