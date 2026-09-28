import logging
import multiprocessing as mp
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .body.behaviors import Body
from .config import settings
from .mind.memory import make_memory
from .providers import build_providers
from .session import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.providers = build_providers()
    app.state.memory = make_memory(settings.memory, settings.memory_path)
    app.state.sim_q, proc = None, None
    if settings.sim and os.path.exists(settings.urdf_path):
        from .body.sim import run_sim
        ctx = mp.get_context("spawn")
        app.state.sim_q = ctx.Queue(maxsize=4)
        proc = ctx.Process(target=run_sim, args=(app.state.sim_q, settings.urdf_path), daemon=True)
        proc.start()
    yield
    if proc:
        proc.terminate()


app = FastAPI(lifespan=lifespan)
app.mount("/client", StaticFiles(directory="client", html=True), name="client")
app.mount("/robot", StaticFiles(directory="robot"), name="robot")  # URDF + mesh for the browser's 3D view


@app.get("/")
def root():
    return RedirectResponse("/client/")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    body = Body(settings.urdf_path, ws.app.state.sim_q)
    await Session(ws, ws.app.state.providers, ws.app.state.memory, body).run()
