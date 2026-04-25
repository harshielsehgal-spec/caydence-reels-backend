from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from reels_router import router as reels_router
from rubric_router import router as rubric_router
from recorded_router import router as recorded_router
from skeleton_router import router as skeleton_router

app = FastAPI(title="Caydence Reels Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters: reels_router defines jobs/reference_cache dicts that
# rubric_router imports and shares. Keep reels_router first.
app.include_router(reels_router)
app.include_router(rubric_router)
app.include_router(recorded_router)
app.include_router(skeleton_router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "caydence-reels-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)