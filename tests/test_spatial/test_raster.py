import math

import numpy as np
import pytest

from src.common.schemas import EntityClass
from src.spatial.raster import (
    LAND_COVER_FRICTION,
    compute_friction_surface,
    land_cover_multiplier,
    tobler_seconds_per_meter,
    tobler_speed_kmh,
    vehicle_slope_impedance_mask,
)


class TestToblerHikingFunction:
    def test_negative_downhill_faster_than_uphill(self):
        """|tan(theta) + 0.05| is asymmetric for signed slope (downhill vs uphill)."""
        uphill = float(tobler_speed_kmh(10.0))
        downhill = float(tobler_speed_kmh(-10.0))
        assert downhill > uphill
        assert uphill > 0 and downhill > 0

    def test_zero_slope_reference_speed(self):
        # v = 6 * exp(-3.5 * |tan(0) + 0.05|) = 6 * exp(-0.175)
        expected = 6.0 * math.exp(-3.5 * 0.05)
        assert math.isclose(float(tobler_speed_kmh(0.0)), expected, rel_tol=1e-9)

    def test_steep_slope_decays_speed(self):
        flat = float(tobler_speed_kmh(0.0))
        steep = float(tobler_speed_kmh(35.0))
        assert steep < flat
        assert steep > 0

    def test_impedance_increases_with_slope(self):
        flat = float(tobler_seconds_per_meter(0.0))
        uphill = float(tobler_seconds_per_meter(20.0))
        assert uphill > flat

    def test_impedance_units_seconds_per_meter(self):
        v_kmh = float(tobler_speed_kmh(0.0))
        imp = float(tobler_seconds_per_meter(0.0))
        assert math.isclose(imp, 3.6 / v_kmh, rel_tol=1e-9)


class TestVehicleSlopeBarrier:
    def test_car_threshold_18_degrees(self):
        slope = np.array([17.9, 18.1])
        mask = vehicle_slope_impedance_mask(slope, travel_mode="car")
        assert mask[0] is np.False_
        assert mask[1] is np.True_

    def test_truck_threshold_15_degrees(self):
        slope = np.array([14.9, 15.1])
        mask = vehicle_slope_impedance_mask(slope, travel_mode="truck")
        assert mask[0] is np.False_
        assert mask[1] is np.True_

    def test_friction_surface_vehicle_steep_slope_infinite(self):
        elev = np.zeros((3, 3))
        slope = np.full((3, 3), 20.0)
        friction = compute_friction_surface(
            elev,
            slope,
            land_cover="road_graded",
            travel_mode="vehicle",
        )
        assert np.all(np.isinf(friction))

    def test_friction_surface_vehicle_off_road_infinite(self):
        elev = np.zeros((2, 2))
        slope = np.zeros((2, 2))
        friction = compute_friction_surface(
            elev,
            slope,
            land_cover="scrub",
            travel_mode="vehicle",
        )
        assert np.all(np.isinf(friction))


class TestLandCoverMultipliers:
    def test_spec_multipliers(self):
        assert land_cover_multiplier("road_graded") == LAND_COVER_FRICTION["road_graded"]
        assert land_cover_multiplier("two_track") == 1.8
        assert land_cover_multiplier("pasture") == 2.5
        assert land_cover_multiplier("scrub") == 4.5
        assert land_cover_multiplier("timber_rock") == 9.0
        assert math.isinf(land_cover_multiplier("barrier"))

    def test_foot_friction_scales_with_land_cover(self):
        elev = np.zeros((1, 1))
        slope = np.zeros((1, 1))
        road = compute_friction_surface(elev, slope, land_cover="road_graded", travel_mode="foot")
        scrub = compute_friction_surface(elev, slope, land_cover="scrub", travel_mode="foot")
        assert float(scrub[0, 0]) > float(road[0, 0])
