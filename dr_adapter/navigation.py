"""Routes over directed exits actually reported by the game client."""
from collections import deque
from .action_space import MOVES

class ObservedMap:
    def __init__(self):
        self.rooms = {}

    def observe(self, room):
        if not isinstance(room, dict) or not isinstance(room.get('id'), str): return
        exits = room.get('exits')
        if not isinstance(exits, dict): return
        self.rooms[room['id']] = {direction: destination for direction, destination in exits.items()
                                  if direction in MOVES and isinstance(destination, str)}

    def next_step(self, source, target, *, explore=False):
        if source == target: return None
        queue = deque([(source, None)])
        seen = {source}
        frontier = None
        while queue:
            room, first = queue.popleft()
            for direction, destination in self.rooms.get(room, {}).items():
                step = first or direction
                if destination == target: return step
                if destination not in self.rooms and frontier is None: frontier = step
                if destination not in seen:
                    seen.add(destination)
                    queue.append((destination, step))
        return frontier if explore else None
