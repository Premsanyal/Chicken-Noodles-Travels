import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def run_accommodation_search(destination, check_in, check_out, budget_per_person, people):
    print(f"🏨 Accommodation Bot: Searching stays in {destination} for {people} people...")

    # 1. Calculate Total Budget for Stay (Assuming 40% of total budget goes to stay)
    total_budget = int(budget_per_person) * int(people)
    stay_budget = total_budget * 0.40 

    # 2. Create the Prompt for the AI
    prompt = f"""
    Act as a travel agent. Find 3 accommodation options in {destination} 
    for dates {check_in} to {check_out}.
    
    Constraints:
    - Total Budget for stay: ₹{stay_budget} approx.
    - Number of people: {people}
    
    Return ONLY valid JSON in this exact format (do not add markdown formatting):
    [
        {{
            "name": "Hotel Name",
            "type": "Hotel/Resort/Homestay",
            "price_per_night": "Price in INR",
            "rating": "4.5/5",
            "description": "Short description matching the vibe."
        }}
    ]
    """

    try:
        # 3. Call the AI Model
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # 4. Clean the response (Remove ```json markers if AI adds them)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_text)

    except Exception as e:
        print(f"⚠️ AI Error: {e}. Using Fallback Data.")
        # Fallback data if AI fails or API key is missing
        return [
            {
                "name": f"Standard {destination} Inn",
                "type": "Hotel",
                "price_per_night": f"₹{int(stay_budget/3)}",
                "rating": "4.0/5",
                "description": "A comfortable fallback option near the city center."
            }
        ]