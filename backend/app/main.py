from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.ingestion import router as ingestion_router
from app.api.query import router as query_router


class HealthResponse(BaseModel):
    status: str


app = FastAPI(
    title="Enterprise Knowledge & Action Agent",
    description="Enterprise AI system foundation API",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingestion_router)



@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}