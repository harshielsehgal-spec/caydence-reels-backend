import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from reels_router import router as reels_router

app = FastAPI(title="Caydence Reels Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://caydence-reels.lovable.app", "*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reels_router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "caydence-reels-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8002)), reload=False)