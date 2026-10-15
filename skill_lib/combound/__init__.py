# skill_lib/compound/__init__.py

from skill_lib.combound.select_obj_by_text import SelectObjectByTextSkill
from skill_lib.combound.find_obj_by_text import FindObjectByTextSkill
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.combound.inspect_medicine_candidates import InspectMedicineCandidatesSkill
from skill_lib.combound.place_to_container import PlaceToContainerSkill
from skill_lib.combound.pick_medicine_and_place_to_container import (
    PickMedicineAndPlaceToContainerSkill,
)


COMPOUND_SKILLS = [
    SelectObjectByTextSkill,
    FindObjectByTextSkill,
    FindAndGraspMedicineSkill,
    InspectMedicineCandidatesSkill,
    PlaceToContainerSkill,
    PickMedicineAndPlaceToContainerSkill,
]
