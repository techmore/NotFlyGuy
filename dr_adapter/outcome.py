"""Read delayed game outcomes without issuing another command."""
import time


def settle_outcome(transport, initial, *, until, deadline, target=2,
                   clock=time.time, sleep=time.sleep):
    state = initial
    while not transport.stopped:
        player = state.get('player') if isinstance(state, dict) else None
        if not isinstance(player, dict) or not isinstance(player.get('hp'), (int, float)):
            transport.stop()
            break
        if player['hp'] <= 0 or player.get('circle', 0) >= target: break
        remaining = min(until, deadline) - clock()
        if remaining <= 0: break
        sleep(min(.5, remaining))
        state = transport.read_state()
    return state
