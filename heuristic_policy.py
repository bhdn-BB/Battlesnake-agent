
DIRECTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def point(cell: dict) -> tuple[int, int]:
    return cell["x"], cell["y"]


def choose_move(state: dict) -> str:

    board = state["board"]
    you = state["you"]

    x, y = point(you["body"][0])

    # Keeping tails occupied is conservative; this is not a full rules simulator.
    occupied = {point(c) for s in board["snakes"] for c in s["body"]}
    occupied.update(point(c) for c in you["body"])

    hazards = {point(c) for c in board.get("hazards", [])}
    food = [point(c) for c in board["food"]]

    threats = {
        (point(s["head"])[0] + dx, point(s["head"])[1] + dy)
        for s in board["snakes"]
        if s["id"] != you["id"] and s["length"] >= you["length"]
        for dx, dy in DIRECTIONS.values()
    }

    candidates = []
    for direction, (dx, dy) in DIRECTIONS.items():
        target = (x + dx, y + dy)

        if not (0 <= target[0] < board["width"] and 0 <= target[1] < board["height"]):
            continue

        if target in occupied:
            continue

        distance = min((abs(target[0] - fx) + abs(target[1] - fy) for fx, fy in food), default=0)
        score = (target not in threats, target not in hazards, -distance)
        candidates.append((score, direction))
    # A valid response is still required when every move looks fatal.
    return max(candidates, key=lambda item: item[0])[1] if candidates else "up"
