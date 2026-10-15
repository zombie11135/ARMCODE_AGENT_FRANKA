# core/skill_registry.py

class SkillRegistry:
    def __init__(self):
        self.skills = {}

    def register(self, skill_cls):
        if not skill_cls.name:
            raise ValueError(f"Skill {skill_cls} has empty name")
        self.skills[skill_cls.name] = skill_cls

    def register_many(self, skill_classes):
        for skill_cls in skill_classes:
            self.register(skill_cls)

    def get(self, name):
        if name not in self.skills:
            raise KeyError(f"Unknown skill: {name}")
        return self.skills[name]

    def list_skills(self):
        return {
            name: {
                "description": cls.description,
                "input_schema": cls.input_schema,
                "output_schema": cls.output_schema,
            }
            for name, cls in self.skills.items()
        }