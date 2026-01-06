import numpy as np

def close_up_to_phase(A, B, atol=1e-8) -> bool:
    ov = np.vdot(A.ravel(), B.ravel())
    phase = 1.0 if abs(ov) < 1e-14 else np.exp(1j * np.angle(ov))
    return np.allclose(A, B * phase, atol=atol)
