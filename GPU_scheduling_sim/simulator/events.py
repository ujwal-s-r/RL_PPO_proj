"""Discrete-event structures and priority queue for cluster simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Optional
import heapq
from simulator.job import Job


class EventType(str, Enum):
    JOB_ARRIVAL = "JOB_ARRIVAL"
    JOB_COMPLETION = "JOB_COMPLETION"
    SCHEDULING_POINT = "SCHEDULING_POINT"
    HORIZON_REACHED = "HORIZON_REACHED"


class EventPriority(IntEnum):
    """Tiebreak priority when event timestamps are identical."""
    COMPLETION = 0  # Process completions first to free resources
    ARRIVAL = 1     # Then process arrivals
    SCHEDULING = 2  # Then make scheduling decisions
    HORIZON = 3     # Horizon check last


@dataclass(order=True)
class Event:
    """Discrete event in the simulation engine."""
    timestamp: float
    priority: int = field(default=EventPriority.SCHEDULING)
    event_type: EventType = field(compare=False, default=EventType.SCHEDULING_POINT)
    job: Optional[Job] = field(compare=False, default=None)
    node_id: Optional[int] = field(compare=False, default=None)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class EventQueue:
    """Min-heap event queue ordered by (timestamp, priority)."""

    def __init__(self) -> None:
        self._heap: list[Event] = []

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def push(self, event: Event) -> None:
        """Schedule an event into the future timeline."""
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        """Pop the earliest chronological event."""
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        """Inspect earliest event without popping."""
        return self._heap[0] if self._heap else None

    def clear(self) -> None:
        """Clear all pending events."""
        self._heap.clear()
