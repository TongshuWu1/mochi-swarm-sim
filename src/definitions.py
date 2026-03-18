from enum import IntEnum, auto

SERVO = "motors_servo"
THRUST_RIGHT = "motor_right_thrust"
THRUST_LEFT = "motor_left_thrust"
ULTRASONIC = "ultrasonic"
AXLE = "motors_axle"
IMU_POS = "imu_pos"
IMU_LIN_VEL = "imu_lin_vel"
IMU_ANG_VEL = "imu_ang_vel"
IMU_QUAT = "imu_quat"


class Action(IntEnum):
    ARMED = 0
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    AUTO_MODE = auto()


class State(IntEnum):
    Z_ALTITUDE = 0
    Z_ALTITUDE_VEL = auto()
    X_ROLL = auto()
    Y_PITCH = auto()
    Z_YAW = auto()
    X_ROLL_RATE = auto()
    Y_PITCH_RATE = auto()
    Z_YAW_RATE = auto()
    NUM_STATES = auto()
