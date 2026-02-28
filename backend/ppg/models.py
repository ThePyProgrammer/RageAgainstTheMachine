from pydantic import BaseModel
from typing import Optional


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class PulseDetectionResponse(BaseModel):
    faceDetected: bool
    face: Optional[Box] = None
    forehead: Optional[Box] = None
    bpm: Optional[float] = None
    faceLocked: Optional[bool] = None
    calibrating: Optional[bool] = None
    calibrationProgress: Optional[int] = None
    calibrationTarget: Optional[int] = None
    waitTime: Optional[float] = None
    bvp: Optional[float] = None
    spo2: Optional[float] = None
    rr: Optional[float] = None
