from ranking_algorithm import TherapistRanker
from config import DB_CONFIG

def test_ranking():
    ranker = TherapistRanker(DB_CONFIG)
    
    # Test search preferences
    preferences = {
        'preferred_therapy_type': 'CBT',
        'issues': ['Anxiety', 'Depression'],
        'preferred_language': 'English',
        'preferred_gender': 'no preference',
        'budget_min': 100,
        'budget_max': 200,
        'cultural_background_preference': 'no preference',
        'language_importance': 8,
        'budget_importance': 7,
        'specialization_importance': 10,
        'experience_importance': 6,
        'gender_importance': 3,
        'cultural_importance': 2,
        'therapy_style_importance': 9,
        'rating_importance': 8
    }
    
    print("🔍 Searching for therapists with preferences:")
    print(f"   Issues: {preferences['issues']}")
    print(f"   Therapy type: {preferences['preferred_therapy_type']}")
    print(f"   Budget: ${preferences['budget_min']}-${preferences['budget_max']}")
    print(f"   Language: {preferences['preferred_language']}\n")
    
    # Get rankings
    results = ranker.rank_therapists(preferences)
    
    # Save search
    search_id = ranker.save_search_and_rankings(preferences, results)
    
    print(f"✅ Search saved with ID: {search_id}\n")
    print("=" * 80)
    print("TOP MATCHES:")
    print("=" * 80 + "\n")
    
    # Display top 5 results
    for i, result in enumerate(results[:5], 1):
        therapist = result['therapist']
        print(f"#{i} - {therapist['name']}")
        print(f"     Title: {therapist['title']}")
        print(f"     Match Score: {result['match_score']}%")
        print(f"     Experience: {therapist['years_experience']} years")
        print(f"     Price: ${therapist['pricing_min']}-${therapist['pricing_max']}")
        print(f"     Specializations: {', '.join(therapist['specializations'] or ['None'])}")
        print(f"     Therapy Styles: {', '.join(therapist['therapy_styles'] or ['None'])}")
        print(f"     Languages: {', '.join(therapist['languages'] or ['None'])}")
        print(f"     Score Breakdown:")
        for key, value in result['score_breakdown'].items():
            print(f"       - {key}: {value}%")
        print()

if __name__ == "__main__":
    test_ranking()