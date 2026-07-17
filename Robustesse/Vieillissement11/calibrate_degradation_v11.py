"""Recalage reproductible des termes PEMWE transitoires sur Rakousky Table 2."""

from __future__ import annotations

from scipy.optimize import least_squares

from Common import degradation_v11 as D


PROTOCOLS = {
    "B": (lambda t: 2.0, 0.1, 194.0),
    "C": (lambda t: 2.0 if t % 12.0 < 6.0 else 1.0, 0.1, 65.0),
    "D": (lambda t: 2.0 if t % 12.0 < 6.0 else 0.0, 0.1, 16.0),
    "E": (lambda t: 2.0 if t % (1.0 / 3.0) < 1.0 / 6.0 else 0.0,
          1.0 / 60.0, 50.0),
}


def replay(pattern, dt_h, duration_h=1009.0):
    state = D.new_ely_state()
    previous = pattern(0.0)
    t = 0.0
    while t < duration_h - 1e-12:
        step = min(dt_h, duration_h - t)
        current = pattern(t)
        state = D.advance_ely_density(state, current, previous, step)
        previous = current
        t += step
    return D.total_uv(state) / duration_h


def residuals(x):
    q, reversible, k0, k1 = x
    D.ELY_V11.update(
        breakin_q_uvph=q, reversible_2_uvph=reversible,
        recovery_0_per_h=k0, recovery_1_per_h=k1,
    )
    return [replay(pattern, dt_h) - target
            for pattern, dt_h, target in PROTOCOLS.values()]


def main():
    fit = least_squares(
        residuals, [60.0, 175.0, 100.0, 0.002],
        bounds=([0.0, 0.0, 0.01, 0.0], [500.0, 500.0, 1000.0, 0.1]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    print("q,b,k0,k1 =", tuple(float(v) for v in fit.x))
    print("residuals =", tuple(float(v) for v in fit.fun))
    for name, (pattern, dt_h, target) in PROTOCOLS.items():
        print(name, replay(pattern, dt_h), "target", target)


if __name__ == "__main__":
    main()
