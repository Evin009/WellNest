import psycopg2
from config import DB_CONFIG

def seed_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    print("Adding sample data...")
    
    # Add specializations
    specializations = [
        'Anxiety', 'Depression', 'Trauma', 'PTSD', 
        'Relationships', 'Grief', 'Stress', 'Self-esteem',
        'Addiction', 'Family Issues', 'Anger Management'
    ]
    for spec in specializations:
        cursor.execute(
            "INSERT INTO specializations (name, category) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (spec, spec.lower())
        )
    print(f"✓ Added {len(specializations)} specializations")
    
    # Add therapy styles
    styles = ['CBT', 'Psychodynamic', 'Humanistic', 'DBT', 'EMDR', 'ACT', 'Family Therapy', 'Gestalt']
    for style in styles:
        cursor.execute(
            "INSERT INTO therapy_styles (name) VALUES (%s) ON CONFLICT DO NOTHING",
            (style,)
        )
    print(f"✓ Added {len(styles)} therapy styles")
    
    # Add languages
    languages = ['English', 'Spanish', 'French', 'Mandarin', 'Arabic', 'German', 'Portuguese']
    for lang in languages:
        cursor.execute(
            "INSERT INTO languages (name) VALUES (%s) ON CONFLICT DO NOTHING",
            (lang,)
        )
    print(f"✓ Added {len(languages)} languages")
    
    # Add sample therapists
    therapists_data = [
        {
            'name': 'Dr. Sarah Johnson',
            'email': 'sarah.johnson@theramatch.com',
            'title': 'Licensed Clinical Psychologist',
            'bio': 'Specializing in anxiety, depression, and trauma with 12 years of experience using evidence-based CBT and EMDR approaches.',
            'years_experience': 12,
            'gender_identity': 'Female',
            'cultural_background': 'Caucasian',
            'pricing_min': 120.00,
            'pricing_max': 180.00,
            'specializations': ['Anxiety', 'Depression', 'Trauma'],
            'therapy_styles': ['CBT', 'EMDR'],
            'languages': ['English', 'Spanish']
        },
        {
            'name': 'Michael Chen, LMFT',
            'email': 'michael.chen@theramatch.com',
            'title': 'Licensed Marriage and Family Therapist',
            'bio': 'Helping couples and families navigate relationship challenges with 8 years of compassionate, solution-focused therapy.',
            'years_experience': 8,
            'gender_identity': 'Male',
            'cultural_background': 'Asian',
            'pricing_min': 100.00,
            'pricing_max': 150.00,
            'specializations': ['Relationships', 'Anxiety', 'Stress'],
            'therapy_styles': ['CBT', 'Humanistic', 'Family Therapy'],
            'languages': ['English', 'Mandarin']
        },
        {
            'name': 'Dr. Lisa Martinez',
            'email': 'lisa.martinez@theramatch.com',
            'title': 'Clinical Psychologist, PhD',
            'bio': 'Specialized in trauma recovery and PTSD treatment using EMDR and somatic approaches for 15 years.',
            'years_experience': 15,
            'gender_identity': 'Female',
            'cultural_background': 'Hispanic',
            'pricing_min': 150.00,
            'pricing_max': 220.00,
            'specializations': ['Trauma', 'PTSD', 'Anxiety'],
            'therapy_styles': ['EMDR', 'Psychodynamic'],
            'languages': ['English', 'Spanish']
        },
        {
            'name': 'James Williams, LCSW',
            'email': 'james.williams@theramatch.com',
            'title': 'Licensed Clinical Social Worker',
            'bio': 'Supporting individuals through life transitions, grief, and depression with person-centered therapy.',
            'years_experience': 6,
            'gender_identity': 'Male',
            'cultural_background': 'African American',
            'pricing_min': 90.00,
            'pricing_max': 130.00,
            'specializations': ['Depression', 'Grief', 'Stress'],
            'therapy_styles': ['Humanistic', 'ACT'],
            'languages': ['English']
        },
        {
            'name': 'Dr. Aisha Patel',
            'email': 'aisha.patel@theramatch.com',
            'title': 'Clinical Psychologist',
            'bio': 'Integrating mindfulness and DBT to help clients with emotion regulation and self-esteem.',
            'years_experience': 10,
            'gender_identity': 'Female',
            'cultural_background': 'South Asian',
            'pricing_min': 110.00,
            'pricing_max': 160.00,
            'specializations': ['Anxiety', 'Self-esteem', 'Stress'],
            'therapy_styles': ['DBT', 'CBT', 'ACT'],
            'languages': ['English', 'Arabic']
        }
    ]
    
    for therapist_data in therapists_data:
        # Insert therapist
        cursor.execute("""
            INSERT INTO therapists (name, email, title, bio, years_experience, 
                                   gender_identity, cultural_background, pricing_min, pricing_max)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            therapist_data['name'],
            therapist_data['email'],
            therapist_data['title'],
            therapist_data['bio'],
            therapist_data['years_experience'],
            therapist_data['gender_identity'],
            therapist_data['cultural_background'],
            therapist_data['pricing_min'],
            therapist_data['pricing_max']
        ))
        
        therapist_id = cursor.fetchone()[0]
        
        # Link specializations
        for spec in therapist_data['specializations']:
            cursor.execute("SELECT id FROM specializations WHERE name = %s", (spec,))
            spec_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO therapist_specializations VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (therapist_id, spec_id)
            )
        
        # Link therapy styles
        for style in therapist_data['therapy_styles']:
            cursor.execute("SELECT id FROM therapy_styles WHERE name = %s", (style,))
            style_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO therapist_therapy_styles VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (therapist_id, style_id)
            )
        
        # Link languages
        for lang in therapist_data['languages']:
            cursor.execute("SELECT id FROM languages WHERE name = %s", (lang,))
            lang_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO therapist_languages VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (therapist_id, lang_id)
            )
        
        print(f"✓ Added therapist: {therapist_data['name']}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✅ Database seeded successfully!")
    print(f"   - {len(therapists_data)} therapists added")
    print(f"   - {len(specializations)} specializations")
    print(f"   - {len(styles)} therapy styles")
    print(f"   - {len(languages)} languages")

if __name__ == "__main__":
    seed_database()