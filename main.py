# type: ignore[reportAttributeAccessIssue]

import sys
import mujoco as mj
from src.simulation import Simulation
from src.controller import Controller

MODEL_XML_PATH = "models/mochi_BlimpRace_balloon_course.xml"


def main():
    model_xml_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_XML_PATH

    model = mj.MjModel.from_xml_path(model_xml_path)
    data = mj.MjData(model)

    controller = Controller(model, data)
    sim = Simulation(model, data, controller)
    sim.run()


if __name__ == "__main__":
    main()
