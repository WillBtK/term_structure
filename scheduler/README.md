# /scheduler — placeholder (not built in Phase 1)

Will hold the estimation-cadence logic described in the top-level `README.md`:
weekly/monthly state-space re-estimation, daily Kalman filter-forward with fixed
parameters, warm-start from the previous parameter vector, and fallback to the
last valid parameter set on optimization failure.

Not built until the US plain-AFNS validation against the SF Fed benchmark passes.
