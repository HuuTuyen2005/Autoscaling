import math

# ==============================
# SYSTEM CAPACITY
# ==============================
CAPACITY_REQ = 100        # req/min / instance
CAPACITY_BYTES = 5e6     # bytes/min / instance

# ==============================
# AUTOSCALING CONFIG
# ==============================
MIN_INSTANCES = 1
MAX_INSTANCES = 10
COOLDOWN = 3              # minutes
SAFETY_MARGIN = 1.15      # predictive buffer


class PredictiveAutoscaler:
    """
    Stateful predictive autoscaler
    (replicates logic from 05_autoscaling_simulation.ipynb)
    """

    def __init__(self, init_instances=1):
        self.instances = init_instances
        self.cooldown = 0

    def step(
        self,
        req_forecast,
        bytes_forecast,
        req_actual,
        bytes_actual,
    ):
        event = "HOLD"

        # ==============================
        # Required instances (FORECAST)
        # ==============================
        required_instances = max(
            math.ceil(req_forecast / CAPACITY_REQ),
            math.ceil(bytes_forecast / CAPACITY_BYTES),
            MIN_INSTANCES,
        )
        required_instances = min(required_instances, MAX_INSTANCES)

        # ==============================
        # SCALE UP (immediate)
        # ==============================
        if required_instances > self.instances:
            event = "SCALE_UP"
            self.instances = required_instances
            self.cooldown = 0

        # ==============================
        # SCALE DOWN (conservative)
        # ==============================
        elif required_instances < self.instances and self.cooldown == 0:
            event = "SCALE_DOWN"
            self.instances -= 1
            self.cooldown = COOLDOWN

        # ==============================
        # SLA CHECK (ACTUAL load)
        # ==============================
        capacity_req = self.instances * CAPACITY_REQ * SAFETY_MARGIN
        capacity_bytes = self.instances * CAPACITY_BYTES * SAFETY_MARGIN

        overload = (
            req_actual > capacity_req
            or bytes_actual > capacity_bytes
        )

        # cooldown tick
        self.cooldown = max(0, self.cooldown - 1)

        return {
            "instances": self.instances,
            "event": event,
            "overload": overload,
            "capacity_req": capacity_req,
            "capacity_bytes": capacity_bytes,
        }
