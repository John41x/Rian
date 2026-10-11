"""Kinematic tracking parameters from specs/00 and specs/03."""

from src.common.schemas import EntityClass

TAU_ASSOCIATE: float = 0.5
SIGMA_T_SECONDS: float = 120.0
ISOCHRONE_CELL_SIZE_M: float = 10.0
STALE_THRESHOLD_SECONDS: float = 30.0 * 60.0
WORKER_INTERVAL_SECONDS: float = 15.0

CLASS_HEADING_KAPPA: dict[EntityClass, float] = {
    EntityClass.PERSON: 1.5,
    EntityClass.CAR: 4.0,
    EntityClass.TRUCK: 4.0,
    EntityClass.COW: 1.0,
    EntityClass.HORSE: 1.2,
    EntityClass.DOG: 1.0,
}

# Low cross-probability for similar vehicle classes (spec §3.2).
VISUAL_SIMILAR_CLASS_PROB: float = 0.15
