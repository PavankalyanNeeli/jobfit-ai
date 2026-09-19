"""
SQLite database layer for the JobFit AI project.

═══════════════════════════════════════════════════════════════
 JobFitDB — Database operations and analytics
═══════════════════════════════════════════════════════════════
"""
import sqlite3
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional

from src.config import DB_PATH, HISTORY_LIMIT
from src.features.skills import SKILL_CATALOG, SKILL_TO_ROLES


class JobFitDB:
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the database connection."""
        self.db_path = db_path or DB_PATH
        # Ensure parent directories exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def initialize(self):
        """Creates all 5 tables if not exists."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            required_skills TEXT,       
            preferred_skills TEXT,      
            min_experience REAL,
            company TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            title TEXT,
            raw_text TEXT,
            skills TEXT,               
            experience_years REAL,
            education_level TEXT,
            target_role TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            difficulty TEXT,
            related_roles TEXT          
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_skills (
            resume_id INTEGER REFERENCES resumes(id),
            skill_id INTEGER REFERENCES skills(id),
            PRIMARY KEY (resume_id, skill_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER REFERENCES resumes(id),
            job_id INTEGER REFERENCES jobs(id),
            score REAL,
            model_version TEXT,
            shap_values TEXT,           
            top_matching_skills TEXT,   
            top_missing_skills TEXT,    
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        self.conn.commit()

    # ── Insert Methods ────────────────────────────────────────────

    def insert_job(self, job_dict: Dict[str, Any]) -> int:
        """Insert a single job and return its id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (
                id, title, description, required_skills, preferred_skills,
                min_experience, company, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_dict.get('id'),
            job_dict.get('title'),
            job_dict.get('description'),
            json.dumps(job_dict.get('required_skills', [])),
            json.dumps(job_dict.get('preferred_skills', [])),
            job_dict.get('min_experience'),
            job_dict.get('company'),
            job_dict.get('source')
        ))
        self.conn.commit()
        return cursor.lastrowid if job_dict.get('id') is None else job_dict['id']

    def insert_resume(self, resume_dict: Dict[str, Any]) -> int:
        """Insert a single resume and return its id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO resumes (
                id, name, title, raw_text, skills, experience_years,
                education_level, target_role, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resume_dict.get('id'),
            resume_dict.get('name'),
            resume_dict.get('title'),
            resume_dict.get('raw_text'),
            json.dumps(resume_dict.get('skills', [])),
            resume_dict.get('experience_years'),
            resume_dict.get('education_level'),
            resume_dict.get('target_role'),
            resume_dict.get('source')
        ))
        self.conn.commit()
        return cursor.lastrowid if resume_dict.get('id') is None else resume_dict['id']

    def insert_skill(self, name: str, category: str, difficulty: str, roles: List[str]) -> int:
        """Insert a skill and return its id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO skills (name, category, difficulty, related_roles)
            VALUES (?, ?, ?, ?)
        """, (name, category, difficulty, json.dumps(list(roles))))
        self.conn.commit()
        
        cursor.execute("SELECT id FROM skills WHERE name = ?", (name,))
        result = cursor.fetchone()
        return result['id']

    def insert_resume_skill(self, resume_id: int, skill_id: int):
        """Link a resume to a skill."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO resume_skills (resume_id, skill_id)
            VALUES (?, ?)
        """, (resume_id, skill_id))
        self.conn.commit()

    def insert_prediction(self, pred_dict: Dict[str, Any]) -> int:
        """Insert a prediction and return its id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                resume_id, job_id, score, model_version,
                shap_values, top_matching_skills, top_missing_skills
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_dict.get('resume_id'),
            pred_dict.get('job_id'),
            pred_dict.get('score'),
            pred_dict.get('model_version'),
            json.dumps(pred_dict.get('shap_values', {})),
            json.dumps(pred_dict.get('top_matching_skills', [])),
            json.dumps(pred_dict.get('top_missing_skills', []))
        ))
        self.conn.commit()
        return cursor.lastrowid

    def bulk_insert_jobs(self, jobs_df: pd.DataFrame):
        """Bulk insert jobs from a DataFrame."""
        for _, row in jobs_df.iterrows():
            self.insert_job(row.to_dict())

    def bulk_insert_resumes(self, resumes_df: pd.DataFrame):
        """Bulk insert resumes from a DataFrame."""
        for _, row in resumes_df.iterrows():
            self.insert_resume(row.to_dict())

    def populate_skills_table(self):
        """Loads all skills from skills.py SKILL_CATALOG."""
        for skill_name, metadata in SKILL_CATALOG.items():
            roles = list(SKILL_TO_ROLES.get(skill_name, set()))
            self.insert_skill(
                name=skill_name,
                category=metadata.get("category", "unknown"),
                difficulty=metadata.get("difficulty", "unknown"),
                roles=roles
            )

    # ── 10 SQL Queries ────────────────────────────────────────────

    def top_skills_for_role(self, role: str, limit: int = 10) -> List[Dict[str, Any]]:
        """1. Top skills for a given role by frequency across resumes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT s.name, COUNT(*) as freq
            FROM resume_skills rs
            JOIN skills s ON rs.skill_id = s.id
            JOIN resumes r ON rs.resume_id = r.id
            WHERE r.target_role = ?
            GROUP BY s.name
            ORDER BY freq DESC
            LIMIT ?
        """, (role, limit))
        return [dict(row) for row in cursor.fetchall()]

    def avg_score_by_role(self) -> List[Dict[str, Any]]:
        """2. Average prediction score grouped by target role."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT r.target_role, AVG(p.score) as avg_score
            FROM predictions p
            JOIN resumes r ON p.resume_id = r.id
            GROUP BY r.target_role
            ORDER BY avg_score DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def most_common_missing_skills(self, limit: int = 10) -> List[Dict[str, Any]]:
        """3. Most frequent missing skills across all predictions."""
        # Using json_each if available in SQLite, otherwise falling back to python
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT j.value as skill, COUNT(*) as freq
                FROM predictions, json_each(predictions.top_missing_skills) as j
                GROUP BY j.value
                ORDER BY freq DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Fallback if json_each is not available
            cursor.execute("SELECT top_missing_skills FROM predictions")
            skill_counts = {}
            for row in cursor.fetchall():
                skills = json.loads(row['top_missing_skills'] or '[]')
                for skill in skills:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [{"skill": k, "freq": v} for k, v in sorted_skills]

    def prediction_history(self, resume_id: int) -> List[Dict[str, Any]]:
        """4. All predictions for a specific resume."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE resume_id = ? 
            ORDER BY created_at DESC
        """, (resume_id,))
        return [dict(row) for row in cursor.fetchall()]

    def jobs_by_avg_score(self, limit: int = 10) -> List[Dict[str, Any]]:
        """5. Jobs ranked by average prediction score."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT j.title, j.company, AVG(p.score) as avg_score
            FROM predictions p
            JOIN jobs j ON p.job_id = j.id
            GROUP BY j.id
            ORDER BY avg_score DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def skill_frequency(self) -> List[Dict[str, Any]]:
        """6. Count of each skill across all resumes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT s.name, COUNT(*) as freq
            FROM resume_skills rs
            JOIN skills s ON rs.skill_id = s.id
            GROUP BY s.name
            ORDER BY freq DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def recent_predictions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """7. Recent N predictions (uses config.HISTORY_LIMIT by default)."""
        limit = limit or HISTORY_LIMIT
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def resumes_above_threshold(self, min_score: float) -> List[Dict[str, Any]]:
        """8. Resumes with any prediction above threshold."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT r.*
            FROM resumes r
            JOIN predictions p ON r.id = p.resume_id
            WHERE p.score >= ?
        """, (min_score,))
        return [dict(row) for row in cursor.fetchall()]

    def skill_coverage_stats(self) -> Dict[str, Any]:
        """9. Skill coverage percentage distribution stats."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT top_matching_skills, top_missing_skills FROM predictions")
        coverages = []
        for row in cursor.fetchall():
            matching = len(json.loads(row['top_matching_skills'] or '[]'))
            missing = len(json.loads(row['top_missing_skills'] or '[]'))
            total = matching + missing
            if total > 0:
                coverages.append(matching / total)
        
        if not coverages:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
            
        import statistics
        return {
            "count": len(coverages),
            "mean": statistics.mean(coverages),
            "min": min(coverages),
            "max": max(coverages)
        }

    def model_comparison(self) -> List[Dict[str, Any]]:
        """10. Average score by model_version."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT model_version, AVG(score) as avg_score, COUNT(*) as num_predictions
            FROM predictions
            GROUP BY model_version
            ORDER BY avg_score DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


if __name__ == '__main__':
    # 1. Creates the DB (in-memory for testing, or a test file)
    test_db_path = Path("test_jobfit.db")
    if test_db_path.exists():
        test_db_path.unlink()
        
    with JobFitDB(test_db_path) as db:
        db.initialize()
        print("Database initialized.")
        
        # 2. Populates the skills table
        db.populate_skills_table()
        print("Skills table populated.")
        
        # Insert some dummy data to ensure queries run
        job_id = db.insert_job({"title": "Data Scientist", "required_skills": ["python", "sql"]})
        resume_id = db.insert_resume({"name": "Alice", "target_role": "data scientist"})
        skill_id = db.insert_skill("python", "programming", "intermediate", ["data scientist"])
        db.insert_resume_skill(resume_id, skill_id)
        db.insert_prediction({
            "resume_id": resume_id, 
            "job_id": job_id, 
            "score": 0.85, 
            "model_version": "v1.0",
            "top_matching_skills": ["python"],
            "top_missing_skills": ["sql"]
        })
        
        # 3. Runs each of the 10 queries and prints results
        print("\n--- Running Queries ---")
        print("1. top_skills_for_role:", db.top_skills_for_role("data scientist"))
        print("2. avg_score_by_role:", db.avg_score_by_role())
        print("3. most_common_missing_skills:", db.most_common_missing_skills())
        print("4. prediction_history:", db.prediction_history(resume_id))
        print("5. jobs_by_avg_score:", db.jobs_by_avg_score())
        print("6. skill_frequency:", db.skill_frequency())
        print("7. recent_predictions:", db.recent_predictions())
        print("8. resumes_above_threshold:", db.resumes_above_threshold(0.8))
        print("9. skill_coverage_stats:", db.skill_coverage_stats())
        print("10. model_comparison:", db.model_comparison())
        
        # 4. Success message
        print("\nAll 10 queries executed successfully")

    # Clean up test DB
    if test_db_path.exists():
        test_db_path.unlink()
