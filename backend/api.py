"""
WellNest REST API
Handles client searches and feedback submission
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from ranking_algorithm import TherapistRanker  # Import the ranker class

# --- App & CORS ---
app = Flask(__name__)
CORS(app)  # or CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- DB config (read from env with sensible defaults) ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "wellnest"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "wellNext123"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

ranker = TherapistRanker(DB_CONFIG)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# ---------- Health ----------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# ---------- Search & Rank ----------
@app.route("/api/therapists/search", methods=["POST"])
def search_therapists():
    """
    Expected JSON body:
    {
        "preferred_therapy_type": "CBT",
        "issues": ["anxiety", "depression"],
        "preferred_language": "English",
        "preferred_gender": "no preference",
        "budget_min": 100,
        "budget_max": 200,
        "cultural_background_preference": "no preference",
        "language_importance": 8,
        "budget_importance": 7,
        "specialization_importance": 10,
        "experience_importance": 6,
        "gender_importance": 3,
        "cultural_importance": 2,
        "therapy_style_importance": 9,
        "rating_importance": 8,
        "client_email": "optional@email.com"
    }
    """
    try:
        preferences = request.get_json(force=True) or {}
        for field in ["issues", "specialization_importance"]:
            if field not in preferences:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        rankings = ranker.rank_therapists(preferences)
        search_id = ranker.save_search_and_rankings(preferences, rankings)

        return jsonify({
            "success": True,
            "search_id": search_id,
            "total_results": len(rankings),
            "results": rankings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Filter options for UI ----------
@app.route("/api/filters", methods=["GET"])
def get_filter_options():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT name FROM specializations ORDER BY name")
        specializations = [r["name"] for r in cur.fetchall()]

        cur.execute("SELECT name FROM therapy_styles ORDER BY name")
        therapy_styles = [r["name"] for r in cur.fetchall()]

        cur.execute("SELECT name FROM languages ORDER BY name")
        languages = [r["name"] for r in cur.fetchall()]

        cur.close(); conn.close()
        return jsonify({
            "success": True,
            "filters": {
                "specializations": specializations,
                "therapy_styles": therapy_styles,
                "languages": languages,
                "gender_options": ["Male", "Female", "Non-binary", "No preference"],
                "cultural_backgrounds": [
                    "Hispanic/Latino", "African American", "Asian",
                    "Caucasian", "Middle Eastern", "No preference"
                ]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Stats (for dashboards/ML sanity checks) ----------
@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT 
                COUNT(DISTINCT cs.id) AS total_searches,
                COUNT(DISTINCT fsf.id) AS total_feedback,
                COUNT(DISTINCT CASE WHEN mh.was_selected = TRUE THEN mh.id END) AS successful_matches,
                AVG(fsf.match_accuracy) AS avg_match_accuracy,
                AVG((fsf.comfort_level + fsf.felt_heard + fsf.clear_communication + 
                     fsf.professional_setting + fsf.would_continue) / 5.0) AS avg_session_rating
            FROM client_searches cs
            LEFT JOIN match_history mh ON cs.id = mh.client_search_id
            LEFT JOIN first_session_feedback fsf ON mh.feedback_id = fsf.id
        """)
        stats = cur.fetchone() or {}
        cur.close(); conn.close()
        return jsonify({"success": True, "statistics": dict(stats)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Therapist details ----------
@app.route("/api/therapists/<int:therapist_id>", methods=["GET"])
def get_therapist_details(therapist_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT 
                t.*,
                ARRAY_AGG(DISTINCT s.name)  FILTER (WHERE s.name IS NOT NULL)  AS specializations,
                ARRAY_AGG(DISTINCT ts.name) FILTER (WHERE ts.name IS NOT NULL) AS therapy_styles,
                ARRAY_AGG(DISTINCT l.name)  FILTER (WHERE l.name IS NOT NULL)  AS languages
            FROM therapists t
            LEFT JOIN therapist_specializations tsp ON t.id = tsp.therapist_id
            LEFT JOIN specializations s           ON tsp.specialization_id = s.id
            LEFT JOIN therapist_therapy_styles tts ON t.id = tts.therapist_id
            LEFT JOIN therapy_styles ts           ON tts.therapy_style_id = ts.id
            LEFT JOIN therapist_languages tl      ON t.id = tl.therapist_id
            LEFT JOIN languages l                 ON tl.language_id = l.id
            WHERE t.id = %s
            GROUP BY t.id
        """, (therapist_id,))
        therapist = cur.fetchone()
        if not therapist:
            cur.close(); conn.close()
            return jsonify({"error": "Therapist not found"}), 404

        cur.execute("""
            SELECT 
                comfort_level, felt_heard, clear_communication,
                professional_setting, would_continue,
                what_worked_well, created_at
            FROM first_session_feedback
            WHERE therapist_id = %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (therapist_id,))
        feedback = cur.fetchall()
        cur.close(); conn.close()
        return jsonify({
            "success": True,
            "therapist": dict(therapist),
            "recent_feedback": [dict(f) for f in feedback]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Submit feedback ----------
@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Expected JSON body with required integer ratings (1-5) and therapist_id.
    """
    try:
        feedback = request.get_json(force=True) or {}

        # Required presence
        required = [
            "therapist_id", "comfort_level", "felt_heard",
            "clear_communication", "professional_setting", "would_continue"
        ]
        for f in required:
            if f not in feedback:
                return jsonify({"error": f"Missing required field: {f}"}), 400

        # Validate rating ranges
        rating_fields = [
            "comfort_level", "felt_heard", "clear_communication",
            "professional_setting", "would_continue", "match_accuracy"
        ]
        for f in rating_fields:
            if f in feedback and feedback[f] is not None:
                v = feedback[f]
                if not isinstance(v, int) or v < 1 or v > 5:
                    return jsonify({"error": f"{f} must be an integer between 1 and 5"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO first_session_feedback (
                therapist_id, client_search_id, comfort_level, felt_heard,
                clear_communication, professional_setting, would_continue,
                what_worked_well, what_could_improve, match_accuracy
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            feedback["therapist_id"],
            feedback.get("client_search_id"),
            feedback["comfort_level"],
            feedback["felt_heard"],
            feedback["clear_communication"],
            feedback["professional_setting"],
            feedback["would_continue"],
            feedback.get("what_worked_well"),
            feedback.get("what_could_improve"),
            feedback.get("match_accuracy"),
        ))
        feedback_id = cur.fetchone()[0]

        # Link to match_history if we know the search
        if feedback.get("client_search_id"):
            cur.execute("""
                UPDATE match_history
                SET feedback_id = %s, was_selected = TRUE
                WHERE client_search_id = %s AND therapist_id = %s
            """, (feedback_id, feedback["client_search_id"], feedback["therapist_id"]))

        conn.commit()
        cur.close(); conn.close()
        return jsonify({
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully. Thank you for helping improve WellNest!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ---------- Entrypoint ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("API_PORT", "5000")))
