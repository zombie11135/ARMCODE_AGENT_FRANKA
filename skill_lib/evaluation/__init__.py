from skill_lib.evaluation.verify_medicine_grasp_vdm import VerifyMedicineGraspVDMSkill
from skill_lib.evaluation.verify_medicine_ocr import VerifyMedicineOCRSkill
from skill_lib.evaluation.verify_medicine_placed_vdm import VerifyMedicinePlacedVDMSkill


EVALUATION_SKILLS = [
    VerifyMedicineOCRSkill,
    VerifyMedicineGraspVDMSkill,
    VerifyMedicinePlacedVDMSkill,
]
