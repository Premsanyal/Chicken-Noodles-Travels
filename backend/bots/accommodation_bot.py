import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def calc_nights(check_in, check_out):
    start = datetime.fromisoformat(check_in)
    end = datetime.fromisoformat(check_out)
    return max((end - start).days, 1)

def run_accommodation_search(destination, check_in, check_out, budget_per_person, people):
    print(f" Accommodation Bot: Searching stays in {destination}...")

    nights = calc_nights(check_in, check_out)

    # Total Budget (40% stays — aligned with budget rules)
    stay_budget = int(budget_per_person) * int(people) * 0.40

    prompt = f"""
    You are an expert hotel planner. Provide the best 3 accommodation options 
    in {destination} for {people} people from {check_in} to {check_out}. 

    Constraints:
    - Total stay budget: ₹{stay_budget}
    - Nights: {nights}
    - Include good places for food nearby
    - Family & friends safe

    STRICT OUTPUT FORMAT:
    {{
      "destination": "{destination}",
      "nights": {nights},
      "stay_budget": {int(stay_budget)},
      "options": [
        {{
          "name": "Hotel Name",
          "type": "Hotel / Resort / Homestay",
          "price_per_night": 0,
          "total_cost": 0,
          "rating": "4.5/5",
          "distance_to_center_km": 0.0,
          "food_nearby": [
              "Example Restaurant 1 - ₹200",
              "Example Restaurant 2 - ₹300"
          ],
          "pros": ["Safe area", "Near attractions"],
          "cons": ["Higher cost in weekends"],
          "fits_budget": true
        }}
      ]
    }}

    Ensure numerical values are provided where needed.
    No markdown, no ```JSON blocks — only clean JSON.
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Clean Gemini response
        clean_json = response.text.replace("```json", "").replace("```", "").strip()

        data = json.loads(clean_json)

        # Safety check
        if "options" not in data:
            raise ValueError("Invalid AI response structure")

        print(" Accommodation Bot Done ✔")
        return data

    except Exception as e:
        print(f" Accommodation Bot failed: {e}")
        # Fallback Minimal Valid Structure
        return {
            "destination": destination,
            "nights": nights,
            "stay_budget": int(stay_budget),
            "options": [
                {
                    "name": "Fallback Lodge",
                    "type": "Budget Stay",
                    "price_per_night": stay_budget / nights,
                    "total_cost": stay_budget,
                    "rating": "4.0/5",
                    "distance_to_center_km": 1.5,
                    "food_nearby": ["Local Veg Restaurant - ₹200"],
                    "pros": ["Cheapest & Convenient"],
                    "cons": ["Less luxury"],
                    "fits_budget": True
                }
            ]
        }
