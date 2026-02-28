from pydantic import BaseModel
from typing import Optional


class Box(BaseModel):
    """Bounding box for face/eye regions."""

    x: int
    y: int
    w: int
    h: int


class PupilData(BaseModel):
    """Pupil detection data"""

    diameter: float
    center_x: float
    center_y: float
    outline_confidence: float
    confidence: Optional[float] = None
    major_axis: float
    minor_axis: float
    angle: float
    circumference: Optional[float] = None
    iris_center_x: Optional[float] = None
    iris_center_y: Optional[float] = None
    iris_radius: Optional[float] = None


class EyeData(BaseModel):
    """Data for a single eye."""

    detected: bool
    region: Optional[Box] = None
    pupil: Optional[PupilData] = None


class PupillometryResponse(BaseModel):
    """Response model for pupillometry detection."""

    faceDetected: bool
    face: Optional[Box] = None
    leftEye: Optional[EyeData] = None
    rightEye: Optional[EyeData] = None
    averagePupilDiameter: Optional[float] = None
    calibrating: Optional[bool] = None
    calibrationProgress: Optional[int] = None
    calibrationTarget: Optional[int] = None
