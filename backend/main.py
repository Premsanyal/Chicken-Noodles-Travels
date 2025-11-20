from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Import your bots
from bots.destination_bot import run_destination_research
from bots.transport_bot import run_transport_planner  # Assuming you created this
from bots.accommodation_bot import run_accommodation_search  # <--- IMPORTED HERE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TravelRequest(BaseModel):
    source: str
    destination: str
    startDate: str
    endDate: str
    people: int
    budget: int
    vehicle: str

@app.post("/api/generate-plan")
async def generate_plan(request: TravelRequest):
    print(f"🚀 Starting Travel Agents for: {request.destination}")

    # --- STEP 1: DESTINATION BOT ---
    # We run this first to get the exact location details (e.g., User types 'Coorg', Bot clarifies 'Madikeri')
    # Note: Since we are calling synchronous functions inside async, we wrap them if needed,
    # but for this tutorial, we will call them directly or assume they are fast.

    dest_data = await run_destination_research(
        request.destination, 
        request.startDate, 
        request.endDate
    )

    # --- STEP 2: PARALLEL EXECUTION (Transport & Accommodation) ---
    # The Accommodation Bot needs the Destination name, but it can run
    # AT THE SAME TIME as the Transport bot to save time.

    loop = asyncio.get_event_loop()

    transport_task = loop.run_in_executor(
        None, 
        run_transport_planner,
        request.source,
        request.destination,
        request.vehicle
    )

    accommodation_task = loop.run_in_executor(
        None,
        run_accommodation_search,
        request.destination,
        request.startDate,
        request.endDate,
        request.budget,
        request.people
    )

    transport_data, accommodation_data = await asyncio.gather(
        transport_task,
        accommodation_task
    )

    # --- STEP 3: BUDGET OPTIMIZER (ALIGNMENT CHECK) ---
    # Simple logic for now: just bundle all three outputs.

    return {
        "message": "Itinerary Generated Successfully",
        "data": {
            "destination_analysis": dest_data,
            "transport_plan": transport_data,
            "accommodation_options": accommodation_data
        }
    }
