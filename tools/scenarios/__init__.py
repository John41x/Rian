"""Automated validation scenarios for the simulation harness."""

from tools.scenarios.vehicle_intrusion import run as run_vehicle_intrusion
from tools.scenarios.pedestrian_hike import run as run_pedestrian_hike
from tools.scenarios.multi_target_gating import run as run_multi_target_gating

SCENARIOS = {
    "vehicle_intrusion": run_vehicle_intrusion,
    "pedestrian_hike": run_pedestrian_hike,
    "multi_target_gating": run_multi_target_gating,
}
