-- PostgreSQL Database Schema for wellNest

-- Therapists Table
CREATE TABLE therapists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(100),
    bio TEXT,
    profile_photo_url TEXT,
    years_experience INTEGER,
    gender_identity VARCHAR(50),
    cultural_background VARCHAR(100),
    pricing_min DECIMAL(10, 2),
    pricing_max DECIMAL(10, 2),
    avg_first_session_rating DECIMAL(3, 2) DEFAULT 0.00,
    total_ratings INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Therapist Specializations (many-to-many)
CREATE TABLE specializations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50) -- e.g., 'anxiety', 'depression', 'trauma', 'relationships'
);

CREATE TABLE therapist_specializations (
    therapist_id INTEGER REFERENCES therapists(id) ON DELETE CASCADE,
    specialization_id INTEGER REFERENCES specializations(id) ON DELETE CASCADE,
    PRIMARY KEY (therapist_id, specialization_id)
);

-- Therapy Styles
CREATE TABLE therapy_styles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL -- e.g., 'CBT', 'Psychodynamic', 'Humanistic'
);

CREATE TABLE therapist_therapy_styles (
    therapist_id INTEGER REFERENCES therapists(id) ON DELETE CASCADE,
    therapy_style_id INTEGER REFERENCES therapy_styles(id) ON DELETE CASCADE,
    PRIMARY KEY (therapist_id, therapy_style_id)
);

-- Languages
CREATE TABLE languages (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE therapist_languages (
    therapist_id INTEGER REFERENCES therapists(id) ON DELETE CASCADE,
    language_id INTEGER REFERENCES languages(id) ON DELETE CASCADE,
    PRIMARY KEY (therapist_id, language_id)
);

-- Client Preferences (stored when they search)
CREATE TABLE client_searches (
    id SERIAL PRIMARY KEY,
    client_email VARCHAR(255), -- optional, for returning users
    preferred_therapy_type VARCHAR(100),
    issues TEXT[], -- Array of issues
    preferred_language VARCHAR(50),
    preferred_gender VARCHAR(50),
    budget_min DECIMAL(10, 2),
    budget_max DECIMAL(10, 2),
    cultural_background_preference VARCHAR(100),
    -- Importance weights (1-10 scale)
    language_importance INTEGER DEFAULT 5,
    budget_importance INTEGER DEFAULT 5,
    specialization_importance INTEGER DEFAULT 10,
    experience_importance INTEGER DEFAULT 5,
    gender_importance INTEGER DEFAULT 3,
    cultural_importance INTEGER DEFAULT 3,
    therapy_style_importance INTEGER DEFAULT 8,
    rating_importance INTEGER DEFAULT 7,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- First Session Feedback
CREATE TABLE first_session_feedback (
    id SERIAL PRIMARY KEY,
    therapist_id INTEGER REFERENCES therapists(id) ON DELETE CASCADE,
    client_search_id INTEGER REFERENCES client_searches(id),
    -- Rating questions (1-5 scale)
    comfort_level INTEGER CHECK (comfort_level BETWEEN 1 AND 5),
    felt_heard INTEGER CHECK (felt_heard BETWEEN 1 AND 5),
    clear_communication INTEGER CHECK (clear_communication BETWEEN 1 AND 5),
    professional_setting INTEGER CHECK (professional_setting BETWEEN 1 AND 5),
    would_continue INTEGER CHECK (would_continue BETWEEN 1 AND 5),
    -- Additional feedback
    what_worked_well TEXT,
    what_could_improve TEXT,
    match_accuracy INTEGER CHECK (match_accuracy BETWEEN 1 AND 5), -- How well did our ranking match their experience?
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match History (for ML training later)
CREATE TABLE match_history (
    id SERIAL PRIMARY KEY,
    client_search_id INTEGER REFERENCES client_searches(id),
    therapist_id INTEGER REFERENCES therapists(id) ON DELETE CASCADE,
    rank_position INTEGER, -- What position was this therapist in the ranked list?
    match_score DECIMAL(5, 2), -- The calculated match score
    was_selected BOOLEAN DEFAULT FALSE, -- Did client book with this therapist?
    feedback_id INTEGER REFERENCES first_session_feedback(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_therapists_rating ON therapists(avg_first_session_rating DESC);
CREATE INDEX idx_therapists_pricing ON therapists(pricing_min, pricing_max);
CREATE INDEX idx_feedback_therapist ON first_session_feedback(therapist_id);
CREATE INDEX idx_match_history_search ON match_history(client_search_id);

-- Trigger to update therapist rating when new feedback is added
CREATE OR REPLACE FUNCTION update_therapist_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE therapists
    SET 
        avg_first_session_rating = (
            SELECT AVG((comfort_level + felt_heard + clear_communication + 
                       professional_setting + would_continue) / 5.0)
            FROM first_session_feedback
            WHERE therapist_id = NEW.therapist_id
        ),
        total_ratings = (
            SELECT COUNT(*)
            FROM first_session_feedback
            WHERE therapist_id = NEW.therapist_id
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.therapist_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_rating
AFTER INSERT ON first_session_feedback
FOR EACH ROW
EXECUTE FUNCTION update_therapist_rating();