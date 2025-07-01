from fastapi import FastAPI
from files import router

app = FastAPI(
    title="Support Assistant File Upload API",
    version="1.0.0"
)

app.include_router(router, prefix="/api")
