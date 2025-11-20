from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the budget bot app
from bots import budget_bot

app = FastAPI(title="Chicken Noodles Travels - Bots Gateway")

# Mount the budget bot as a sub-application
app.mount("/budget", budget_bot.app)

# CORS (frontend will need this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bots.main:app", host="0.0.0.0", port=8004, reload=True)
