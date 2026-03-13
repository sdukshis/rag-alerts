import json
from pathlib import Path
from typing import List

from rag_alerts.models import Incident


def load_incidents(path: Path) -> List[Incident]:
    incidents: List[Incident] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            incidents.append(Incident(**payload))
    return incidents
