"""
TheraMatch Machine Learning Training Pipeline
Learns from feedback to improve ranking algorithm over time
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
from sqlalchemy import create_engine
from datetime import datetime

# Suppress warnings for small datasets
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TherapistMatchingML:
    def __init__(self, db_config):
        self.db_config = db_config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_metadata = {}

    # ---------------------------------------------------------------------
    def extract_training_data(self):
        """
        Extract historical match data with outcomes for training
        """
        print("Connecting to database...")
        engine_url = (
            f"postgresql+psycopg2://{self.db_config['user']}:"
            f"{self.db_config['password']}@{self.db_config['host']}:"
            f"5432/{self.db_config['database']}"
        )
        engine = create_engine(engine_url)

        query = """
            SELECT 
                cs.preferred_therapy_type,
                cs.issues,
                cs.preferred_language,
                cs.preferred_gender,
                cs.budget_min,
                cs.budget_max,
                cs.cultural_background_preference,
                cs.language_importance,
                cs.budget_importance,
                cs.specialization_importance,
                cs.experience_importance,
                cs.gender_importance,
                cs.cultural_importance,
                cs.therapy_style_importance,
                cs.rating_importance,

                mh.rank_position,
                mh.match_score,
                mh.was_selected,

                t.years_experience,
                t.gender_identity,
                t.cultural_background,
                t.pricing_min,
                t.pricing_max,
                t.avg_first_session_rating,
                t.total_ratings,

                ARRAY_AGG(DISTINCT s.name) FILTER (WHERE s.name IS NOT NULL) as specializations,
                ARRAY_AGG(DISTINCT ts.name) FILTER (WHERE ts.name IS NOT NULL) as therapy_styles,
                ARRAY_AGG(DISTINCT l.name) FILTER (WHERE l.name IS NOT NULL) as languages,

                fsf.comfort_level,
                fsf.felt_heard,
                fsf.clear_communication,
                fsf.professional_setting,
                fsf.would_continue,
                fsf.match_accuracy,
                (fsf.comfort_level + fsf.felt_heard + fsf.clear_communication + 
                 fsf.professional_setting + fsf.would_continue) / 5.0 as avg_feedback_score

            FROM match_history mh
            JOIN client_searches cs ON mh.client_search_id = cs.id
            JOIN therapists t ON mh.therapist_id = t.id
            LEFT JOIN first_session_feedback fsf ON mh.feedback_id = fsf.id
            LEFT JOIN therapist_specializations tsp ON t.id = tsp.therapist_id
            LEFT JOIN specializations s ON tsp.specialization_id = s.id
            LEFT JOIN therapist_therapy_styles tts ON t.id = tts.therapist_id
            LEFT JOIN therapy_styles ts ON tts.therapy_style_id = ts.id
            LEFT JOIN therapist_languages tl ON t.id = tl.therapist_id
            LEFT JOIN languages l ON tl.language_id = l.id
            WHERE mh.was_selected = TRUE  
            AND fsf.id IS NOT NULL
            GROUP BY cs.id, mh.id, t.id, fsf.id
        """

        df = pd.read_sql_query(query, engine)
        print(f"   Extracted {len(df)} training samples")
        return df

    # ---------------------------------------------------------------------
    def _calculate_budget_overlap(self, client_min, client_max, therapist_min, therapist_max):
        if pd.isna(client_min) or pd.isna(client_max):
            return 0.5
        overlap_min = max(therapist_min, client_min)
        overlap_max = min(therapist_max, client_max)
        if overlap_min > overlap_max:
            return 0.0
        overlap_size = overlap_max - overlap_min
        client_range = client_max - client_min
        return overlap_size / client_range if client_range > 0 else 1.0

    # ---------------------------------------------------------------------
    def engineer_features(self, df):
        """Feature engineering"""
        df['budget_overlap'] = df.apply(
            lambda row: self._calculate_budget_overlap(
                row['budget_min'], row['budget_max'],
                row['pricing_min'], row['pricing_max']
            ), axis=1
        )

        df['specialization_match_count'] = df.apply(
            lambda row: len(set(row['issues']) & set(row['specializations'] or []))
            if isinstance(row['issues'], list) else 0, axis=1
        )

        df['therapy_style_match'] = df.apply(
            lambda row: 1 if row['preferred_therapy_type'] in (row['therapy_styles'] or []) else 0,
            axis=1
        )

        df['language_match'] = df.apply(
            lambda row: 1 if row['preferred_language'] in (row['languages'] or []) else 0,
            axis=1
        )

        df['gender_match'] = df.apply(
            lambda row: 1 if (row['preferred_gender'] == 'no preference' or
                              row['preferred_gender'] == row['gender_identity']) else 0,
            axis=1
        )

        df['cultural_match'] = df.apply(
            lambda row: 1 if (row['cultural_background_preference'] == 'no preference' or
                              row['cultural_background_preference'] == row['cultural_background']) else 0,
            axis=1
        )

        df['experience_score'] = np.log(df['years_experience'] + 1)
        df['rating_confidence'] = np.minimum(df['total_ratings'] / 20, 1.0)
        df['weighted_rating'] = df['avg_first_session_rating'] * df['rating_confidence']

        df['weighted_specialization'] = df['specialization_match_count'] * df['specialization_importance']
        df['weighted_therapy_style'] = df['therapy_style_match'] * df['therapy_style_importance']
        df['weighted_language'] = df['language_match'] * df['language_importance']
        df['weighted_budget'] = df['budget_overlap'] * df['budget_importance']
        return df

    # ---------------------------------------------------------------------
    def prepare_features(self, df):
        """Select and prepare final feature set"""
        feature_columns = [
            'budget_overlap', 'specialization_match_count', 'therapy_style_match', 'language_match',
            'gender_match', 'cultural_match', 'experience_score', 'weighted_rating', 'rating_confidence',
            'language_importance', 'budget_importance', 'specialization_importance',
            'experience_importance', 'gender_importance', 'cultural_importance',
            'therapy_style_importance', 'rating_importance', 'weighted_specialization',
            'weighted_therapy_style', 'weighted_language', 'weighted_budget',
            'rank_position', 'match_score'
        ]
        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0)
        y = df['avg_feedback_score']
        return X, y

    # ---------------------------------------------------------------------
    def train_model(self, X, y, model_type='gradient_boosting'):
        """Train ranking improvement model"""
        if len(X) < 5:
            print("⚠️  Too few samples for test split — training on full data.")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=42
            )

        print("Training model...")
        self.model.fit(X_train_scaled, y_train)

        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Cross-validation (only if enough samples)
        if len(X_train_scaled) >= 5:
            n_splits = min(5, len(X_train_scaled))
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=n_splits, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
        else:
            cv_rmse = None

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': getattr(self.model, 'feature_importances_', np.zeros(len(self.feature_columns)))
        }).sort_values('importance', ascending=False)

        self.model_metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': model_type,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'cv_rmse': float(cv_rmse) if cv_rmse else None,
            'n_samples': len(X),
            'feature_importance': feature_importance.to_dict('records')
        }

        print("\n=== Model Training Results ===")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Train R²:   {train_r2:.4f}")
        print(f"Test R²:    {test_r2:.4f}")
        if cv_rmse:
            print(f"CV RMSE:    {cv_rmse:.4f}")
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

    # ---------------------------------------------------------------------
    def save_model(self, filepath='theramatch_model.pkl'):
        if self.model is None:
            raise ValueError("No model trained yet!")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.model_metadata
        }, filepath)
        print(f"✅ Model saved to {filepath}")

    # ---------------------------------------------------------------------
    def full_training_pipeline(self):
        print("Starting ML training pipeline...\n")
        df = self.extract_training_data()
        if len(df) == 0:
            print("❌ No data found in database. Please insert feedback first.")
            return None
        if len(df) < 50:
            print("⚠️  Less than 50 samples — training anyway for testing purposes.\n")

        df = self.engineer_features(df)
        X, y = self.prepare_features(df)
        self.train_model(X, y)
        self.save_model()
        print("\n✅ Training pipeline complete!")
        return self.model_metadata


# ---------------------------------------------------------------------
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'wellnest',
        'user': 'postgres',
        'password': 'wellNext123'
    }

    ml_trainer = TherapistMatchingML(db_config)
    results = ml_trainer.full_training_pipeline()

    if results:
        print("\n" + "=" * 50)
        print("Model is ready to enhance rankings!")
        print("=" * 50)
