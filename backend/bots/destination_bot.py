import os
import json
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

async def run_destination_research(destination, start_date, end_date):
    print(f"🌍 Destination Bot: Analyzing {destination} for {start_date}...")

    # 1. The System Prompt (The Brains)
    # We ask for strictly formatted JSON so our Python code doesn't break.
    prompt = f"""
    Act as an expert Travel Planner. I need a detailed analysis for a trip to {destination} 
    from {start_date} to {end_date}.
    
    You must return the response in STRICT JSON format. Do not write any introduction or conclusion. 
    
    The JSON must follow this structure:
    {{
        "official_name": "City, State, Country",
        "coordinates": {{ "lat": 0.0, "lng": 0.0 }}, 
        "weather_forecast": "Short summary of expected weather for these dates.",
        "best_area_to_stay": "Name of the specific area central to attractions (e.g., 'Madikeri Center')",
        "attractions": [
            {{
                "name": "Place Name",
                "type": "Adventure/Sightseeing/Relaxation",
                "time_needed": "2 hours",
                "best_time_to_visit": "Morning/Evening",
                "description": "Short description"
            }},
            {{
                "name": "Place Name 2",
                "type": "Sightseeing",
                "time_needed": "1 hour",
                "best_time_to_visit": "Afternoon",
                "description": "Short description"
            }}
        ]
    }}
    
    Generate 4-5 top attractions that are logically close to each other to minimize travel time.
    """

    try:
        # 2. Call Google Gemini (Async)
        model = genai.GenerativeModel('gemini-pro')
        
        # We run the blocking API call in a separate thread so the server stays fast
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        
        # 3. Clean the Output (Crucial Step)
        # sometimes AI adds ```json at the start, we need to remove it
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        
        destination_data = json.loads(clean_text)
        
        print("✅ Destination Analysis Complete.")
        return destination_data

    except Exception as e:
        print(f"❌ Error in Destination Bot: {e}")
        # Fallback data in case AI fails
        return {
            "official_name": destination,
            "weather_forecast": "Data Unavailable",
            "best_area_to_stay": destination,
            "attractions": []
        }

# --- Quick Test Block (Run this file directly to test) ---
if __name__ == "__main__":
    # This allows you to test just this bot without running the full server
    result = asyncio.run(run_destination_research("Munnar", "2025-11-20", "2025-11-23"))
    print(json.dumps(result, indent=2))