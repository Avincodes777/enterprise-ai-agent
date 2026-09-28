from fastapi import FastAPI

app = FastAPI(
    title="Enterprise Knowledge & Action Agent",
    description="Enterprise AI system foundation API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
