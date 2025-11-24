from datetime import datetime


def get_health_status() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


