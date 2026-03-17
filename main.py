# type: ignore[reportAttributeAccessIssue]

import mujoco as mj
from src.simulation import Simulation
from src.controller import Controller

MODEL_XML_PATH = "models/mochi_BlimpRace.xml"


def main():
    # 1. Load model and data
    model = mj.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mj.MjData(model)

    # 2. Create the controller
    controller = Controller(model, data)

    # 3. Create the simulation "engine" and pass it the controller
    sim = Simulation(model, data, controller)

    # 4. Run the simulation
    sim.run()


if __name__ == "__main__":
    main()
