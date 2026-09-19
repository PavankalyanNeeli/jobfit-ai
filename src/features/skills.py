"""
Skill Knowledge Graph — the DSA hashmap showcase.

Data Structures used (interview-ready):
    - HashMap (dict):  skill → roles, role → skills  — O(1) lookups
    - Set operations:  intersection (match), difference (gap)  — O(min(m,n))
    - Sorting:         skills ranked by relevance/specificity   — O(k log k)

Interview line: "I used hashmaps for O(1) skill resolution and set
operations for O(n) match/gap computation — the same patterns I'd
use in a system design interview, but here they're in production."
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ═══════════════════════════════════════════════════════════════
#  SKILL CATALOG — master metadata for every known skill
# ═══════════════════════════════════════════════════════════════

SKILL_CATALOG: Dict[str, dict] = {
    # ── Programming Languages ─────────────────────────────────
    "python":       {"category": "programming",      "difficulty": "intermediate"},
    "r":            {"category": "programming",      "difficulty": "intermediate"},
    "sql":          {"category": "programming",      "difficulty": "beginner"},
    "java":         {"category": "programming",      "difficulty": "intermediate"},
    "javascript":   {"category": "programming",      "difficulty": "intermediate"},
    "c++":          {"category": "programming",      "difficulty": "advanced"},
    "scala":        {"category": "programming",      "difficulty": "advanced"},
    "go":           {"category": "programming",      "difficulty": "intermediate"},
    "typescript":   {"category": "programming",      "difficulty": "intermediate"},
    "rust":         {"category": "programming",      "difficulty": "advanced"},

    # ── Data Science & ML ─────────────────────────────────────
    "machine learning":          {"category": "ml", "difficulty": "advanced"},
    "deep learning":             {"category": "ml", "difficulty": "advanced"},
    "natural language processing": {"category": "ml", "difficulty": "advanced"},
    "computer vision":           {"category": "ml", "difficulty": "advanced"},
    "statistics":                {"category": "ml", "difficulty": "intermediate"},
    "data analysis":             {"category": "ml", "difficulty": "beginner"},
    "data visualization":        {"category": "ml", "difficulty": "beginner"},
    "feature engineering":       {"category": "ml", "difficulty": "intermediate"},
    "model deployment":          {"category": "ml", "difficulty": "intermediate"},
    "a/b testing":               {"category": "ml", "difficulty": "intermediate"},
    "time series":               {"category": "ml", "difficulty": "intermediate"},
    "recommendation systems":    {"category": "ml", "difficulty": "advanced"},

    # ── Frameworks & Libraries ────────────────────────────────
    "tensorflow":   {"category": "framework", "difficulty": "advanced"},
    "pytorch":      {"category": "framework", "difficulty": "advanced"},
    "scikit-learn": {"category": "framework", "difficulty": "intermediate"},
    "pandas":       {"category": "framework", "difficulty": "beginner"},
    "numpy":        {"category": "framework", "difficulty": "beginner"},
    "spark":        {"category": "framework", "difficulty": "advanced"},
    "hadoop":       {"category": "framework", "difficulty": "advanced"},
    "flask":        {"category": "framework", "difficulty": "intermediate"},
    "django":       {"category": "framework", "difficulty": "intermediate"},
    "react":        {"category": "framework", "difficulty": "intermediate"},
    "streamlit":    {"category": "framework", "difficulty": "beginner"},
    "fastapi":      {"category": "framework", "difficulty": "intermediate"},
    "keras":        {"category": "framework", "difficulty": "intermediate"},

    # ── Cloud & Infrastructure ────────────────────────────────
    "aws":          {"category": "cloud", "difficulty": "intermediate"},
    "gcp":          {"category": "cloud", "difficulty": "intermediate"},
    "azure":        {"category": "cloud", "difficulty": "intermediate"},
    "docker":       {"category": "cloud", "difficulty": "intermediate"},
    "kubernetes":   {"category": "cloud", "difficulty": "advanced"},
    "ci/cd":        {"category": "cloud", "difficulty": "intermediate"},
    "git":          {"category": "cloud", "difficulty": "beginner"},
    "linux":        {"category": "cloud", "difficulty": "intermediate"},
    "terraform":    {"category": "cloud", "difficulty": "advanced"},

    # ── Databases ─────────────────────────────────────────────
    "postgresql":    {"category": "database", "difficulty": "intermediate"},
    "mongodb":       {"category": "database", "difficulty": "intermediate"},
    "redis":         {"category": "database", "difficulty": "intermediate"},
    "elasticsearch": {"category": "database", "difficulty": "intermediate"},
    "mysql":         {"category": "database", "difficulty": "beginner"},
    "cassandra":     {"category": "database", "difficulty": "advanced"},

    # ── Data Engineering ──────────────────────────────────────
    "etl":            {"category": "data_engineering", "difficulty": "intermediate"},
    "data pipelines": {"category": "data_engineering", "difficulty": "intermediate"},
    "airflow":        {"category": "data_engineering", "difficulty": "advanced"},
    "kafka":          {"category": "data_engineering", "difficulty": "advanced"},
    "dbt":            {"category": "data_engineering", "difficulty": "intermediate"},

    # ── Soft Skills ───────────────────────────────────────────
    "communication":      {"category": "soft", "difficulty": "beginner"},
    "leadership":         {"category": "soft", "difficulty": "intermediate"},
    "problem solving":    {"category": "soft", "difficulty": "beginner"},
    "teamwork":           {"category": "soft", "difficulty": "beginner"},
    "project management": {"category": "soft", "difficulty": "intermediate"},
    "agile":              {"category": "soft", "difficulty": "beginner"},
}


# ═══════════════════════════════════════════════════════════════
#  SKILL → ROLES  HashMap  (O(1) lookup per skill)
# ═══════════════════════════════════════════════════════════════

SKILL_TO_ROLES: Dict[str, Set[str]] = {
    # Programming
    "python":     {"data scientist", "ml engineer", "data analyst", "data engineer", "backend developer"},
    "r":          {"data scientist", "data analyst", "research scientist"},
    "sql":        {"data analyst", "data engineer", "data scientist", "backend developer", "business analyst"},
    "java":       {"backend developer", "data engineer", "ml engineer"},
    "javascript": {"frontend developer", "full stack developer"},
    "c++":        {"ml engineer", "research scientist"},
    "scala":      {"data engineer", "ml engineer"},
    "go":         {"backend developer", "devops engineer"},
    "typescript": {"frontend developer", "full stack developer"},
    "rust":       {"backend developer", "ml engineer"},

    # ML / DS
    "machine learning":          {"data scientist", "ml engineer", "research scientist"},
    "deep learning":             {"ml engineer", "data scientist", "research scientist"},
    "natural language processing": {"ml engineer", "data scientist", "research scientist"},
    "computer vision":           {"ml engineer", "data scientist", "research scientist"},
    "statistics":                {"data scientist", "data analyst", "research scientist"},
    "data analysis":             {"data analyst", "business analyst", "data scientist"},
    "data visualization":        {"data analyst", "business analyst", "data scientist"},
    "feature engineering":       {"data scientist", "ml engineer"},
    "model deployment":          {"ml engineer", "devops engineer"},
    "a/b testing":               {"data scientist", "data analyst", "business analyst"},
    "time series":               {"data scientist", "ml engineer", "research scientist"},
    "recommendation systems":    {"ml engineer", "data scientist"},

    # Frameworks
    "tensorflow":   {"ml engineer", "data scientist", "research scientist"},
    "pytorch":      {"ml engineer", "data scientist", "research scientist"},
    "scikit-learn": {"data scientist", "ml engineer", "data analyst"},
    "pandas":       {"data analyst", "data scientist", "data engineer"},
    "numpy":        {"data scientist", "ml engineer", "data analyst"},
    "spark":        {"data engineer", "ml engineer", "data scientist"},
    "hadoop":       {"data engineer"},
    "flask":        {"backend developer", "ml engineer"},
    "django":       {"backend developer", "full stack developer"},
    "react":        {"frontend developer", "full stack developer"},
    "streamlit":    {"data scientist", "ml engineer"},
    "fastapi":      {"backend developer", "ml engineer"},
    "keras":        {"ml engineer", "data scientist"},

    # Cloud
    "aws":        {"devops engineer", "data engineer", "ml engineer"},
    "gcp":        {"devops engineer", "data engineer", "ml engineer"},
    "azure":      {"devops engineer", "data engineer"},
    "docker":     {"devops engineer", "ml engineer", "data engineer", "backend developer"},
    "kubernetes": {"devops engineer", "ml engineer"},
    "ci/cd":      {"devops engineer", "ml engineer", "backend developer"},
    "git":        {"data scientist", "ml engineer", "backend developer", "frontend developer", "data engineer"},
    "linux":      {"devops engineer", "data engineer", "backend developer"},
    "terraform":  {"devops engineer"},

    # Databases
    "postgresql":    {"backend developer", "data engineer", "data analyst"},
    "mongodb":       {"backend developer", "full stack developer"},
    "redis":         {"backend developer", "data engineer"},
    "elasticsearch": {"data engineer", "backend developer"},
    "mysql":         {"backend developer", "data analyst", "full stack developer"},
    "cassandra":     {"data engineer", "backend developer"},

    # Data Engineering
    "etl":            {"data engineer", "data analyst"},
    "data pipelines": {"data engineer", "ml engineer"},
    "airflow":        {"data engineer", "ml engineer"},
    "kafka":          {"data engineer", "backend developer"},
    "dbt":            {"data engineer", "data analyst"},

    # Soft Skills
    "communication":      {"data analyst", "business analyst", "data scientist"},
    "leadership":         {"data scientist", "ml engineer", "data engineer"},
    "problem solving":    {"data scientist", "ml engineer", "backend developer"},
    "teamwork":           {"data analyst", "data scientist", "ml engineer"},
    "project management": {"data scientist", "ml engineer", "data engineer"},
    "agile":              {"backend developer", "frontend developer", "data engineer", "ml engineer"},
}


# ═══════════════════════════════════════════════════════════════
#  ROLE → SKILLS  HashMap  (inverse, built programmatically)
# ═══════════════════════════════════════════════════════════════

ROLE_TO_SKILLS: Dict[str, Set[str]] = defaultdict(set)
for _skill, _roles in SKILL_TO_ROLES.items():
    for _role in _roles:
        ROLE_TO_SKILLS[_role].add(_skill)
ROLE_TO_SKILLS = dict(ROLE_TO_SKILLS)  # freeze


# ═══════════════════════════════════════════════════════════════
#  SKILL ALIASES  (normalization map — O(1) canonicalization)
# ═══════════════════════════════════════════════════════════════

SKILL_ALIASES: Dict[str, str] = {
    "ml":                    "machine learning",
    "dl":                    "deep learning",
    "nlp":                   "natural language processing",
    "cv":                    "computer vision",
    "sklearn":               "scikit-learn",
    "sk-learn":              "scikit-learn",
    "postgres":              "postgresql",
    "mongo":                 "mongodb",
    "k8s":                   "kubernetes",
    "tf":                    "tensorflow",
    "amazon web services":   "aws",
    "google cloud":          "gcp",
    "google cloud platform": "gcp",
    "microsoft azure":       "azure",
    "data viz":              "data visualization",
    "pm":                    "project management",
    "js":                    "javascript",
    "ts":                    "typescript",
    "node":                  "javascript",
    "node.js":               "javascript",
    "react.js":              "react",
    "scikit learn":          "scikit-learn",
    "elastic search":        "elasticsearch",
    "time-series":           "time series",
    "ab testing":            "a/b testing",
    "ab-testing":            "a/b testing",
}


# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def normalize_skill(skill: str) -> str:
    """Normalize a skill string to its canonical form.

    Uses the alias map for O(1) lookup, then falls back to
    lowercase + strip for direct catalog match.
    """
    cleaned = skill.strip().lower()
    return SKILL_ALIASES.get(cleaned, cleaned)


def extract_skills(text: str) -> List[str]:
    """Extract known skills from free text using hashmap lookups.

    Strategy: check each known skill (and its aliases) against the
    lowercased text.  O(k) where k = catalog size, not text length.

    Returns:
        Sorted list of unique canonical skill names found in text.
    """
    text_lower = text.lower()
    found: Set[str] = set()

    # Check all canonical skills against text
    for skill in SKILL_CATALOG:
        if len(skill) <= 2:
            # Short skills (r, go) need word-boundary matching
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found.add(skill)
        else:
            if skill in text_lower:
                found.add(skill)

    # Check aliases
    for alias, canonical in SKILL_ALIASES.items():
        if len(alias) <= 2:
            if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                found.add(canonical)
        else:
            if alias in text_lower:
                found.add(canonical)

    return sorted(found)


def compute_skill_match(
    resume_skills: Set[str],
    job_skills: Set[str],
) -> Tuple[Set[str], Set[str], float]:
    """Compute skill match using set operations.

    Args:
        resume_skills: Skills found in the resume.
        job_skills:    Skills required by the job posting.

    Returns:
        (matched_skills, missing_skills, coverage_ratio)

    Time complexity: O(min(m, n)) for intersection, O(n) for difference.
    """
    matched = resume_skills & job_skills         # intersection
    missing = job_skills - resume_skills         # difference
    coverage = len(matched) / len(job_skills) if job_skills else 0.0
    return matched, missing, coverage


def get_roles_for_skill(skill: str) -> Set[str]:
    """O(1) lookup: which roles need this skill?"""
    canonical = normalize_skill(skill)
    return SKILL_TO_ROLES.get(canonical, set())


def get_skills_for_role(role: str) -> Set[str]:
    """O(1) lookup: which skills does this role need?"""
    return ROLE_TO_SKILLS.get(role.lower(), set())


def get_skill_category(skill: str) -> str:
    """O(1) lookup: what category is this skill in?"""
    canonical = normalize_skill(skill)
    info = SKILL_CATALOG.get(canonical)
    return info["category"] if info else "unknown"


def get_skill_difficulty(skill: str) -> str:
    """O(1) lookup: what difficulty is this skill?"""
    canonical = normalize_skill(skill)
    info = SKILL_CATALOG.get(canonical)
    return info["difficulty"] if info else "unknown"


def rank_skills_by_relevance(
    skills: List[str],
    target_role: str,
) -> List[Tuple[str, float]]:
    """Rank skills by relevance to a target role.

    Relevance = inverse of how many roles share that skill.
    A skill shared by fewer roles is more distinctive / relevant.

    Returns:
        List of (skill, relevance_score) sorted descending by score.
    """
    role_skills = get_skills_for_role(target_role)
    scored: list[Tuple[str, float]] = []

    for skill in skills:
        canonical = normalize_skill(skill)
        if canonical in role_skills:
            roles_with_skill = SKILL_TO_ROLES.get(canonical, set())
            # Fewer roles sharing this skill → higher specificity
            relevance = 1.0 / len(roles_with_skill) if roles_with_skill else 0.0
            scored.append((canonical, round(relevance, 4)))

    # Sort by relevance descending — deliberate algorithmic choice
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def get_all_skills() -> List[str]:
    """Return all canonical skill names, sorted."""
    return sorted(SKILL_CATALOG.keys())


def get_all_roles() -> List[str]:
    """Return all known roles, sorted."""
    return sorted(ROLE_TO_SKILLS.keys())
