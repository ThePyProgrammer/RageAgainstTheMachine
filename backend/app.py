import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from eeg.routes import router as eeg_router
# from ppg.routes import router as ppg_router
# from ocular.routes import router as ocular_router
from mi.routes import router as mi_router
from opponent.routes import router as opponent_router

from shared.config.logging import configure_logging, get_logger
# from ppg.controller import initialize as initialize_ppg
from mi.initialization import initialize as initialize_mi

configure_logging()
logger = get_logger("app")


def run_startup_task(name: str, initializer) -> None:
    try:
        logger.info("Initializing %s service...", name)
        initializer()
        logger.info("%s service initialized.", name)
    except Exception:
        logger.exception("Error initializing %s.", name)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting up Thonk...")
    for name, initializer in (
        # ("PPG", initialize_ppg),
        ("Motor Imagery", initialize_mi),
    ):
        run_startup_task(name, initializer)
    yield


app = FastAPI(title="Thonk", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(eeg_router)
# app.include_router(ppg_router)
# app.include_router(ocular_router)
app.include_router(mi_router)
app.include_router(opponent_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
