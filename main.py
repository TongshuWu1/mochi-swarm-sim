# type: ignore[reportAttributeAccessIssue]

import sys
from pathlib import Path

import mujoco as mj

from src.simulation import Simulation
from src.controller import Controller

PROJECT_ROOT = Path(__file__).resolve().parent
TRACK_MODELS = {
    "default": PROJECT_ROOT / "models" / "mochi_BlimpRace_balloon_course.xml",
    "s": PROJECT_ROOT / "models" / "mochi_BlimpRace_balloon_course_s.xml",
    "figure8": PROJECT_ROOT / "models" / "mochi_BlimpRace_balloon_course_figure8.xml",
}
MODEL_XML_PATH = TRACK_MODELS["default"]


def resolve_model_path(arg: str | None) -> Path:
    if arg is None:
        return MODEL_XML_PATH

    key = arg.strip().lower()
    if key in TRACK_MODELS:
        return TRACK_MODELS[key]

    return Path(arg).expanduser()


def main():
    model_xml_path = resolve_model_path(sys.argv[1] if len(sys.argv) > 1 else None)

    model = mj.MjModel.from_xml_path(str(model_xml_path))
    data = mj.MjData(model)

    controller = Controller(model, data)
    sim = Simulation(model, data, controller)
    sim.run()


if __name__ == "__main__":
    main()
