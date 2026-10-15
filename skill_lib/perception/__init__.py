# skill_lib/perception/__init__.py

from skill_lib.perception.capture import (
    CaptureGlobalImageSkill,
    CaptureWristRGBDSkill,
)

from skill_lib.perception.vdm import VDMSkill
from skill_lib.perception.reference_recognition import (
    RegisterObjectReferenceSkill, MatchObjectReferenceSkill, RecognizeObjectSkill,
)


PERCEPTION_SKILLS = [
    CaptureGlobalImageSkill,
    CaptureWristRGBDSkill,
    VDMSkill,
    RegisterObjectReferenceSkill,
    MatchObjectReferenceSkill,
    RecognizeObjectSkill,
]
