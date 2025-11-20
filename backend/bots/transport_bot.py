# transport_planner_v2.py
import os
import uuid
import asyncio
import logging
import random
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Protocol
from abc import abstractmethod
from enum import Enum
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
import httpx
from async_lru import alru_cache  # pip install async-lru

# ------------------------
# 1. Application Configuration
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TransportPlanner")

class Settings(BaseSettings):
    # Keys
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    REVV_API_KEY: Optional[str] = None
    ZOOMCAR_PARTNER_KEY: Optional[str] = None
    
    # Constants
    FUEL_PRICE_PER_L: float = 105.0
    DEFAULT_AVG_SPEED_KMPH: float = 60.0
    CO2_GRAMS_PER_KM: float = 120.0  # Avg car
    
    # Simulation flags
    ENABLE_MOCK_ON_FAIL: bool = True

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# ------------------------
# 2. Domain Models (Pydantic V2)
# ------------------------
class VehicleType(str, Enum):
    HATCHBACK = "hatchback"
    SEDAN = "sedan"
    SUV = "suv"
    MUV = "muv"
    LUXURY = "luxury"

class VehicleSpecs(BaseModel):
    capacity: int
    fuel_economy_l_100km: float
    base_rate: float

# Static Knowledge Base
VEHICLE_SPECS = {
    VehicleType.HATCHBACK: VehicleSpecs(capacity=4, fuel_economy_l_100km=6.0, base_rate=1500.0),
    VehicleType.SEDAN: VehicleSpecs(capacity=4, fuel_economy_l_100km=7.5, base_rate=2200.0),
    VehicleType.SUV: VehicleSpecs(capacity=5, fuel_economy_l_100km=9.0, base_rate=3200.0),
    VehicleType.MUV: VehicleSpecs(capacity=7, fuel_economy_l_100km=10.5, base_rate=4000.0),
    VehicleType.LUXURY: VehicleSpecs(capacity=4, fuel_economy_l_100km=12.0, base_rate=6000.0),
}

class DateRange(BaseModel):
    start: date = Field(..., alias="from")
    end: date = Field(..., alias="to")

    @field_validator("end")
    @classmethod
    def end_not_before_start(cls, v: date, info):
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("'to' date must be equal or after 'from' date")
        return v

class RentalOffer(BaseModel):
    provider: str
    vehicle_type: VehicleType
    name: str
    capacity: int
    total_price: float
    currency: str = "INR"
    daily_rate: float
    rating: Optional[float] = Field(None, ge=0, le=5)
    booking_url: Optional[str] = None
    meta: Dict[str, Any] = {}

class RouteInfo(BaseModel):
    distance_km: float
    duration_minutes: int
    polyline: Optional[str] = None
    start_address: str
    end_address: str
    traffic_level: str = "normal"

class CarbonStats(BaseModel):
    co2_kg: float
    offset_cost_estimate: float  # Cost to offset this carbon

class PlanRequest(BaseModel):
    origin: str
    destination: str
    dates: DateRange
    people: int = Field(..., ge=1, le=20)
    preference: str = Field("balanced", pattern="^(cheapest|fastest|balanced|own_vehicle)$")

class PlanResponse(BaseModel):
    plan_id: str
    summary: Dict[str, Any]
    route: RouteInfo
    carbon_footprint: CarbonStats
    rental_options: List[RentalOffer]
    cost_breakdown: Dict[str, float]

# ------------------------
# 3. External Clients (Adapters)
# ------------------------

class MapsClient:
    """Handles routing logic with Caching"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/directions/json"

    @alru_cache(maxsize=100)
    async def get_route(self, origin: str, dest: str) -> RouteInfo:
        """Cached route fetching"""
        if not self.api_key:
            logger.warning("No Google Maps Key. Using Geometrical Estimate.")
            return self._estimate_route(origin, dest)

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(self.base_url, params={
                    "origin": origin, "destination": dest, "key": self.api_key, "mode": "driving"
                })
                resp.raise_for_status()
                data = resp.json()
                
                if data["status"] != "OK":
                    raise ValueError(f"Maps API Error: {data['status']}")
                
                leg = data["routes"][0]["legs"][0]
                return RouteInfo(
                    distance_km=leg["distance"]["value"] / 1000.0,
                    duration_minutes=int(leg["duration"]["value"] / 60),
                    polyline=data["routes"][0]["overview_polyline"]["points"],
                    start_address=leg["start_address"],
                    end_address=leg["end_address"]
                )
            except Exception as e:
                logger.error(f"Maps API failed: {e}")
                return self._estimate_route(origin, dest)

    def _estimate_route(self, origin: str, dest: str) -> RouteInfo:
        # Fallback logic for demo purposes (randomized slightly for realism)
        dist = random.uniform(50, 400)
        speed = settings.DEFAULT_AVG_SPEED_KMPH
        return RouteInfo(
            distance_km=round(dist, 2),
            duration_minutes=int((dist / speed) * 60),
            start_address=origin,
            end_address=dest,
            traffic_level="estimated"
        )

# ------------------------
# 4. Rental Provider Strategy
# ------------------------

class RentalProvider(Protocol):
    async def search(self, city: str, start: date, end: date, seats: int) -> List[RentalOffer]:
        ...

class RevvProvider:
    def __init__(self, key: Optional[str]):
        self.key = key
        
    async def search(self, city: str, start: date, end: date, seats: int) -> List[RentalOffer]:
        if not self.key: return []
        # ... Real API implementation here ...
        return []

class ZoomcarProvider:
    def __init__(self, key: Optional[str]):
        self.key = key
        
    async def search(self, city: str, start: date, end: date, seats: int) -> List[RentalOffer]:
        if not self.key: return []
        # ... Real API implementation here ...
        return []

class MockSimulationProvider:
    """Generates realistic fake data for demos or fallbacks"""
    async def search(self, city: str, start: date, end: date, seats: int) -> List[RentalOffer]:
        await asyncio.sleep(0.1) # Simulate network latency
        days = (end - start).days + 1
        offers = []
        
        # Determine eligible vehicle types based on seats
        eligible_types = [vt for vt, spec in VEHICLE_SPECS.items() if spec.capacity >= seats]
        
        for vt in eligible_types:
            spec = VEHICLE_SPECS[vt]
            # Add randomness to price
            price_factor = random.uniform(0.9, 1.3)
            daily = round(spec.base_rate * price_factor, -1)
            
            offers.append(RentalOffer(
                provider=random.choice(["Revv", "Zoomcar", "MyChoize"]),
                vehicle_type=vt,
                name=f"{vt.value.title()} Class - {random.choice(['Swift', 'City', 'Creta', 'Innova'])}",
                capacity=spec.capacity,
                daily_rate=daily,
                total_price=daily * days,
                rating=round(random.uniform(3.5, 5.0), 1),
                booking_url="https://example.com/book"
            ))
        return offers

# ------------------------
# 5. Core Logic Service
# ------------------------

class TransportPlannerService:
    def __init__(self):
        self.maps = MapsClient(settings.GOOGLE_MAPS_API_KEY)
        self.providers: List[RentalProvider] = [
            RevvProvider(settings.REVV_API_KEY),
            ZoomcarProvider(settings.ZOOMCAR_PARTNER_KEY)
        ]
        # Always add mock if configured or no keys present
        if settings.ENABLE_MOCK_ON_FAIL:
            self.providers.append(MockSimulationProvider())

    async def get_rental_aggregations(self, city: str, start: date, end: date, people: int) -> List[RentalOffer]:
        # Run all provider searches in parallel
        tasks = [p.search(city, start, end, people) for p in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_offers = []
        for r in results:
            if isinstance(r, list):
                all_offers.extend(r)
        
        # Deduplication logic
        unique = {}
        for o in all_offers:
            key = f"{o.provider}-{o.vehicle_type}-{o.total_price}"
            if key not in unique:
                unique[key] = o
        
        return sorted(list(unique.values()), key=lambda x: x.total_price)

    def calculate_carbon(self, distance_km: float, vehicle_type: str = "sedan") -> CarbonStats:
        # Simple heuristic: 120g/km for sedan. SUV is higher.
        multiplier = 1.0
        if "suv" in vehicle_type or "muv" in vehicle_type: multiplier = 1.3
        
        co2_kg = (distance_km * settings.CO2_GRAMS_PER_KM * multiplier) / 1000.0
        # Carbon credit estimate ~ $20 per ton
        cost = (co2_kg / 1000.0) * 20.0 * 83.0 # Convert to INR
        
        return CarbonStats(co2_kg=round(co2_kg, 2), offset_cost_estimate=round(cost, 2))

    async def generate_plan(self, req: PlanRequest) -> PlanResponse:
        # 1. Parallel execution: Route + Rentals
        route_task = self.maps.get_route(req.origin, req.destination)
        
        rental_task = asyncio.create_task(asyncio.sleep(0)) # no-op default
        if req.preference != "own_vehicle":
            rental_task = self.get_rental_aggregations(req.origin, req.dates.start, req.dates.end, req.people)
        
        # Await both
        route, rentals = await asyncio.gather(route_task, rental_task)
        if isinstance(rentals, list) is False: rentals = [] # safety check

        # 2. Cost Calculation
        fuel_cost = 0.0
        selected_vehicle_cost = 0.0
        
        # Determine baseline vehicle for calculations
        if req.preference == "own_vehicle":
            # Assume user has appropriate car
            v_spec = list(VEHICLE_SPECS.values())[0] # Default
            for v in VEHICLE_SPECS.values():
                if v.capacity >= req.people: 
                    v_spec = v
                    break
            fuel_needed = (route.distance_km * v_spec.fuel_economy_l_100km) / 100.0
            fuel_cost = fuel_needed * settings.FUEL_PRICE_PER_L
            selected_vehicle_cost = fuel_cost # Operational cost only
        else:
            # If rentals exist, pick best based on preference
            if rentals:
                best = rentals[0] # Already sorted by price
                selected_vehicle_cost = best.total_price
                # Fuel is usually extra in rentals, add estimate
                v_spec = VEHICLE_SPECS.get(VehicleType(best.vehicle_type), VEHICLE_SPECS[VehicleType.SEDAN])
                fuel_needed = (route.distance_km * v_spec.fuel_economy_l_100km) / 100.0
                fuel_cost = fuel_needed * settings.FUEL_PRICE_PER_L

        # 3. Carbon Calculation
        carbon = self.calculate_carbon(route.distance_km)

        return PlanResponse(
            plan_id=str(uuid.uuid4()),
            summary={
                "total_distance": route.distance_km,
                "total_duration_hours": round(route.duration_minutes / 60, 1),
                "primary_mode": "Drive (Own)" if req.preference == "own_vehicle" else "Drive (Rental)"
            },
            route=route,
            carbon_footprint=carbon,
            rental_options=rentals[:5], # Return top 5
            cost_breakdown={
                "vehicle_base_cost": round(selected_vehicle_cost, 2),
                "estimated_fuel_cost": round(fuel_cost, 2),
                "toll_estimate": round(route.distance_km * 2.0, 2) # Rough estimate
            }
        )

# ------------------------
# 6. API Application
# ------------------------

app = FastAPI(
    title="Smart Transport Planner",
    description="Generates transport plans with live routing, rental aggregation, and carbon analysis.",
    version="2.0.0"
)

# Dependency Injection
def get_planner_service():
    return TransportPlannerService()

@app.post("/api/v2/plan", response_model=PlanResponse)
async def create_transport_plan(
    request: PlanRequest,
    service: TransportPlannerService = Depends(get_planner_service)
):
    try:
        return await service.generate_plan(request)
    except Exception as e:
        logger.error(f"Planning failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal planning error")

@app.get("/health")
async def health_check():
    return {"status": "active", "env": "production" if not settings.ENABLE_MOCK_ON_FAIL else "simulation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)