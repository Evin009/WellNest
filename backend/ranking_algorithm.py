"""
TheraMatch Therapist Ranking Algorithm
Calculates weighted match scores for therapists based on client preferences.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import numpy as np
from decimal import Decimal


class TherapistRanker:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    def get_connection(self):
        return psycopg2.connect(**self.db_config)

    # ---------- HELPER CONVERSION ----------
    def _to_float(self, value):
        """Safely convert Decimal/None to float"""
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except Exception:
            return 0.0

    # ---------- MATCH CALCULATIONS ----------
    def calculate_specialization_match(self, therapist_specs: List[str], client_issues: List[str]) -> float:
        if not client_issues:
            return 0.5
        
        therapist_set = set(spec.lower() for spec in therapist_specs)
        client_set = set(issue.lower() for issue in client_issues)
        matches = len(therapist_set & client_set)
        if matches == 0:
            return 0.0
        return min(matches / len(client_set), 1.0)

    def calculate_therapy_style_match(self, therapist_styles: List[str], preferred_style: str) -> float:
        if not preferred_style:
            return 0.5
        styles = [s.lower() for s in therapist_styles]
        return 1.0 if preferred_style.lower() in styles else 0.0

    def calculate_language_match(self, therapist_languages: List[str], preferred_language: str) -> float:
        if not preferred_language:
            return 0.5
        langs = [l.lower() for l in therapist_languages]
        return 1.0 if preferred_language.lower() in langs else 0.0

    def calculate_budget_match(self, therapist_min, therapist_max, client_min, client_max) -> float:
        """Calculate budget compatibility safely."""
        if client_min is None or client_max is None:
            return 0.5

        therapist_min = self._to_float(therapist_min)
        therapist_max = self._to_float(therapist_max)
        client_min = self._to_float(client_min)
        client_max = self._to_float(client_max)

        overlap_min = max(therapist_min, client_min)
        overlap_max = min(therapist_max, client_max)

        if overlap_min > overlap_max:
            # No overlap
            if therapist_min > client_max:
                gap = therapist_min - client_max
                max_gap = therapist_min if therapist_min > 0 else 1
                return max(0.0, 1.0 - (gap / max_gap))
            else:
                return 0.8  # therapist cheaper
        else:
            overlap_size = overlap_max - overlap_min
            client_range = max(client_max - client_min, 1e-6)
            return min(1.0, overlap_size / client_range)

    def calculate_gender_match(self, therapist_gender: str, preferred_gender: str) -> float:
        if not preferred_gender or preferred_gender.lower() == "no preference":
            return 1.0
        if therapist_gender and therapist_gender.lower() == preferred_gender.lower():
            return 1.0
        return 0.0

    def calculate_cultural_match(self, therapist_background: str, preferred_background: str) -> float:
        if not preferred_background or preferred_background.lower() == "no preference":
            return 1.0
        if therapist_background and therapist_background.lower() == preferred_background.lower():
            return 1.0
        return 0.3

    def calculate_experience_score(self, years_experience) -> float:
        years_experience = self._to_float(years_experience)
        if years_experience <= 0:
            return 0.5
        return min(1.0, np.log(years_experience + 1) / np.log(21))

    def calculate_rating_score(self, avg_rating, total_ratings) -> float:
        avg_rating = self._to_float(avg_rating)
        total_ratings = self._to_float(total_ratings)

        if avg_rating <= 0:
            return 0.5
        normalized = (avg_rating - 1) / 4.0  # map 1–5 → 0–1
        confidence = min(1.0, total_ratings / 20)
        return normalized * confidence + 0.5 * (1 - confidence)

    # ---------- MAIN RANKING ----------
    def rank_therapists(self, search_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT 
                t.id, t.name, t.title, t.bio, t.profile_photo_url,
                t.years_experience, t.gender_identity, t.cultural_background,
                t.pricing_min, t.pricing_max, t.avg_first_session_rating, t.total_ratings,
                ARRAY_AGG(DISTINCT s.name) FILTER (WHERE s.name IS NOT NULL) AS specializations,
                ARRAY_AGG(DISTINCT ts.name) FILTER (WHERE ts.name IS NOT NULL) AS therapy_styles,
                ARRAY_AGG(DISTINCT l.name) FILTER (WHERE l.name IS NOT NULL) AS languages
            FROM therapists t
            LEFT JOIN therapist_specializations tsp ON t.id = tsp.therapist_id
            LEFT JOIN specializations s ON tsp.specialization_id = s.id
            LEFT JOIN therapist_therapy_styles tts ON t.id = tts.therapist_id
            LEFT JOIN therapy_styles ts ON tts.therapy_style_id = ts.id
            LEFT JOIN therapist_languages tl ON t.id = tl.therapist_id
            LEFT JOIN languages l ON tl.language_id = l.id
            GROUP BY t.id
        """)
        therapists = cursor.fetchall()

        ranked = []
        for t in therapists:
            s = {}
            s["specialization"] = self.calculate_specialization_match(t["specializations"] or [], search_preferences.get("issues", []))
            s["therapy_style"] = self.calculate_therapy_style_match(t["therapy_styles"] or [], search_preferences.get("preferred_therapy_type", ""))
            s["language"] = self.calculate_language_match(t["languages"] or [], search_preferences.get("preferred_language", ""))
            s["budget"] = self.calculate_budget_match(t["pricing_min"], t["pricing_max"], search_preferences.get("budget_min"), search_preferences.get("budget_max"))
            s["gender"] = self.calculate_gender_match(t["gender_identity"], search_preferences.get("preferred_gender", ""))
            s["cultural"] = self.calculate_cultural_match(t["cultural_background"], search_preferences.get("cultural_background_preference", ""))
            s["experience"] = self.calculate_experience_score(t["years_experience"])
            s["rating"] = self.calculate_rating_score(t["avg_first_session_rating"], t["total_ratings"])

            weights = {
                "specialization": search_preferences.get("specialization_importance", 10) / 10,
                "therapy_style": search_preferences.get("therapy_style_importance", 8) / 10,
                "language": search_preferences.get("language_importance", 5) / 10,
                "budget": search_preferences.get("budget_importance", 8) / 10,
                "gender": search_preferences.get("gender_importance", 3) / 10,
                "cultural": search_preferences.get("cultural_importance", 3) / 10,
                "experience": search_preferences.get("experience_importance", 5) / 10,
                "rating": search_preferences.get("rating_importance", 7) / 10
            }

            total_weight = sum(weights.values())
            weighted_sum = sum(s[k] * weights[k] for k in s)
            final_score = (weighted_sum / total_weight) * 100.0

            ranked.append({
                "therapist": dict(t),
                "match_score": round(final_score, 2),
                "score_breakdown": {k: round(v * 100, 1) for k, v in s.items()}
            })

        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        cursor.close()
        conn.close()
        return ranked

    # ---------- SAVE HISTORY ----------
    def save_search_and_rankings(self, prefs: Dict[str, Any], rankings: List[Dict[str, Any]]) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO client_searches (
                client_email, preferred_therapy_type, issues, preferred_language,
                preferred_gender, budget_min, budget_max, cultural_background_preference,
                language_importance, budget_importance, specialization_importance,
                experience_importance, gender_importance, cultural_importance,
                therapy_style_importance, rating_importance
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            prefs.get("client_email"),
            prefs.get("preferred_therapy_type"),
            prefs.get("issues", []),
            prefs.get("preferred_language"),
            prefs.get("preferred_gender"),
            prefs.get("budget_min"),
            prefs.get("budget_max"),
            prefs.get("cultural_background_preference"),
            prefs.get("language_importance", 5),
            prefs.get("budget_importance", 5),
            prefs.get("specialization_importance", 10),
            prefs.get("experience_importance", 5),
            prefs.get("gender_importance", 3),
            prefs.get("cultural_importance", 3),
            prefs.get("therapy_style_importance", 8),
            prefs.get("rating_importance", 7)
        ))

        search_id = cursor.fetchone()[0]

        insert_match = """
            INSERT INTO match_history (
                client_search_id, therapist_id, rank_position, match_score
            ) VALUES (%s, %s, %s, %s)
        """

        for i, r in enumerate(rankings, 1):
            match_score = float(r["match_score"])  # ✅ Convert NumPy float to Python float
            cursor.execute(insert_match, (
                search_id,
                r["therapist"]["id"],
                i,
                match_score
            ))

        conn.commit()
        cursor.close()
        conn.close()
        return search_id

