import copy

def analyze_gaps(resume_text, job_desc, pipeline):
    """
    Computes skill gaps by finding the set difference between JD and resume.
    Then performs ablation testing to find the delta impact of adding each missing skill.
    
    Args:
        resume_text (str): Candidate's resume text.
        job_desc (str): Target job description text.
        pipeline: Trained sklearn pipeline.
        
    Returns:
        list of dict: [{"skill": "Python", "delta": 0.15}, ...]
    """
    from src.features.skills import extract_skills
    
    # 1. Extract skills
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_desc))
    
    # 2. Find missing skills
    missing_skills = job_skills - resume_skills
    
    if not missing_skills:
        return []
    
    # 3. Calculate baseline score
    baseline_X = [resume_text + " | " + job_desc]
    baseline_score = pipeline.predict_proba(baseline_X)[0][1]
    
    gaps = []
    
    # 4. Ablation testing (O(N) where N is missing skills)
    for skill in missing_skills:
        # Simulate adding the skill to the resume
        simulated_resume = resume_text + f" {skill} "
        simulated_X = [simulated_resume + " | " + job_desc]
        
        simulated_score = pipeline.predict_proba(simulated_X)[0][1]
        delta = simulated_score - baseline_score
        
        gaps.append({
            "skill": skill,
            "delta": delta,
            "simulated_score": simulated_score,
            "baseline_score": baseline_score
        })
        
    return gaps
