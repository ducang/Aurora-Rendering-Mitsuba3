import numpy as np
import matplotlib.pyplot as plt

# Jos Stam Fluid Grid Solver
def set_bnd_rect(b, x, h, w):
    x[0,   1:w+1] = x[h,   1:w+1]
    x[h+1, 1:w+1] = x[1,   1:w+1]
    if b == 2:
        x[1:h+1, 0  ] = -x[1:h+1, 1]
        x[1:h+1, w+1] = -x[1:h+1, w]
    else:
        x[1:h+1, 0  ] = x[1:h+1, 1]
        x[1:h+1, w+1] = x[1:h+1, w]
    x[0,   0  ] = 0.5 * (x[1, 0  ] + x[0,   1])
    x[0,   w+1] = 0.5 * (x[1, w+1] + x[0,   w])
    x[h+1, 0  ] = 0.5 * (x[h, 0  ] + x[h+1, 1])
    x[h+1, w+1] = 0.5 * (x[h, w+1] + x[h+1, w])


def advect_rect(b, d, d0, u, v, dt, h, w, i_grid, j_grid):
    x = i_grid - dt * u[1:h+1, 1:w+1]
    y = j_grid - dt * v[1:h+1, 1:w+1]
    x = ((x - 0.5) % h) + 0.5
    y = np.clip(y, 0.5, w + 0.5)
    xf, yf = np.floor(x), np.floor(y)
    i0 = (xf.astype(np.int32) - 1) % h + 1
    i1 = i0 % h + 1
    j0 = np.clip(yf.astype(np.int32), 1, w)
    j1 = np.clip(j0 + 1, 1, w)
    sx, sy = x - xf, y - yf
    d[1:h+1, 1:w+1] = (
        (1 - sx) * ((1 - sy) * d0[i0, j0] + sy * d0[i0, j1]) +
        sx * ((1 - sy) * d0[i1, j0] + sy * d0[i1, j1])
    )
    set_bnd_rect(b, d, h, w)


def project_rect(u, v, p, div, h, w, iters=100):
    div[1:h+1, 1:w+1] = -0.5 * (
        u[2:h+2, 1:w+1] - u[0:h,   1:w+1] +
        v[1:h+1, 2:w+2] - v[1:h+1, 0:w  ]
    )               
    p.fill(0.0)
    set_bnd_rect(0, div, h, w)
    set_bnd_rect(0, p, h, w)
    for _ in range(iters):
        p[1:h+1, 1:w+1] = (
            div[1:h+1, 1:w+1] +
            p[0:h,   1:w+1] + p[2:h+2, 1:w+1] +
            p[1:h+1, 0:w  ] + p[1:h+1, 2:w+2]
        ) * 0.25
        set_bnd_rect(0, p, h, w)
    u[1:h+1, 1:w+1] -= 0.5 * (p[2:h+2, 1:w+1] - p[0:h,   1:w+1]) 
    v[1:h+1, 1:w+1] -= 0.5 * (p[1:h+1, 2:w+2] - p[1:h+1, 0:w  ]) 
    set_bnd_rect(1, u, h, w)
    set_bnd_rect(2, v, h, w)


def vel_step(u, v, u0, v0, dt, h, w, i_grid, j_grid):
    u += dt * u0
    v += dt * v0
    project_rect(u, v, u0, v0, h, w)
    u0, u = u, u0
    v0, v = v, v0
    advect_rect(1, u, u0, u0, v0, dt, h, w, i_grid, j_grid)
    advect_rect(2, v, v0, u0, v0, dt, h, w, i_grid, j_grid)
    project_rect(u, v, u0, v0, h, w)
    return u, v, u0, v0


def dens_step(x, x0, u, v, dt, h, w, i_grid, j_grid):
    x += dt * x0
    x0, x = x, x0
    advect_rect(0, x, x0, u, v, dt, h, w, i_grid, j_grid)
    return x, x0
# Kelvin Helmholtz initialization
def initialize_KH_instability(h, w, seed=0):
    x  = np.zeros((h+2, w+2), dtype=np.float32)
    x0 = np.zeros_like(x)
    u  = np.zeros_like(x)
    v  = np.zeros_like(x)
    u0 = np.zeros_like(x)
    v0 = np.zeros_like(x)

    sigma = 8.0
    sigma_c = 12.0
    sigma_v = 24.0
    U0 = 1.0
    eps = 0.08

    jc = 0.5 * (w + 1)
    ii = np.arange(1, h+1, dtype=np.float32)[:, np.newaxis]
    jj = np.arange(1, w+1, dtype=np.float32)[np.newaxis, :]
    rng = np.random.default_rng(seed)

    band = np.exp(-0.5 * ((jj - jc) / sigma) ** 2)
    x[1:h+1, 1:w+1] = band
    u[1:h+1, 1:w+1] = U0 * np.tanh((jj - jc) / sigma_c)

    modes    = np.arange(2, 14)
    v_perturb = np.zeros((h, w), dtype=np.float32)
    for m in modes:
        phi = rng.uniform(0, 2 * np.pi)
        v_perturb += np.sin(2.0 * np.pi * m * (ii - 1) / h + phi).astype(np.float32)
    v_perturb /= len(modes)
    v[1:h+1, 1:w+1] = eps * v_perturb * np.exp(-0.5 * ((jj - jc) / sigma_v) ** 2)

    set_bnd_rect(0, x, h, w)
    set_bnd_rect(1, u, h, w)
    set_bnd_rect(2, v, h, w)
    return x, x0, u, v, u0, v0

def simulate(h=1024, w=96, seed=0):
    x, x0, u, v, u0, v0 = initialize_KH_instability(h, w, seed)
    num_steps = 500
    dt = 0.8
    i_grid, j_grid = np.meshgrid(
        np.arange(1, h+1, dtype=np.float32),
        np.arange(1, w+1, dtype=np.float32),
        indexing='ij'
    )
    for _ in range(num_steps):
        u0.fill(0.0)
        v0.fill(0.0)
        x0.fill(0.0)
        u, v, u0, v0 = vel_step(u, v, u0, v0, dt, h, w, i_grid, j_grid)
        x, x0 = dens_step(x, x0, u, v, dt, h, w, i_grid, j_grid)

    strip = x[1:h+1, 1:w+1].copy()
    strip -= strip.min()
    if strip.max() > 0:
        strip /= strip.max()
    return strip.astype(np.float32)

# vectorized batch projection
def project_to_centerline_batch(XY, frame):

    P = frame["P"] # (M,   2)
    seg_len = frame["seg_len"]  # (M-1,)
    seg_t = frame["seg_t"] # (M-1, 2)
    seg_n = frame["seg_n"] # (M-1, 2)
    S = frame["S"]  # (M,)

    # rel[n, i] = XY[n] - P[i], shape (N, M-1, 2)
    rel = XY[:, np.newaxis, :] - P[np.newaxis, :-1, :]

    a= np.einsum('nij,ij->ni', rel, seg_t) # (N, M-1)
    tau= np.clip(a / seg_len[np.newaxis, :], 0.0, 1.0) # (N, M-1)

    Q = P[np.newaxis, :-1, :] + tau[:, :, np.newaxis] * seg_len[np.newaxis, :, np.newaxis] * seg_t[np.newaxis, :, :]
    delta = XY[:, np.newaxis, :] - Q # (N, M-1, 2)
    dist2 = np.sum(delta ** 2, axis=2) # (N, M-1)

    best_i = np.argmin(dist2, axis=1) # (N,)
    idx = np.arange(XY.shape[0])
    best_tau = tau[idx, best_i]

    s_world = S[best_i] + best_tau * seg_len[best_i]
    d_world = np.einsum('ni,ni->n', delta[idx, best_i], seg_n[best_i])
    return s_world, d_world


# vectorized rasterize
def rasterize_wrapped_footprint(strip, frame, curtain_width, repeat_length, xmin, xmax, ymin, ymax, out_h=512, out_w=512):
    xs = np.linspace(xmin, xmax, out_w, dtype=np.float32)
    ys = np.linspace(ymin, ymax, out_h, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.stack([XX.ravel(), YY.ravel()], axis=1)

    s_world, d_world = project_to_centerline_batch(XY, frame)

    half_w = 0.5 * curtain_width
    dn = d_world / max(half_w, 1e-8)
    mask = np.abs(dn) <= 1.0

    side_falloff = np.exp(-0.5 * (dn / 0.55) ** 2)

    u = np.clip(s_world / repeat_length, 0.0, 0.9999)
    v = np.clip(0.5 * dn + 0.5, 0.0, 1.0)

    Hs, Hd = strip.shape
    px = u * Hs
    py = v * (Hd - 1)

    x0 = np.floor(px).astype(np.int32).clip(0, Hs - 2)
    x1 = x0 + 1
    y0 = np.clip(np.floor(py).astype(np.int32), 0, Hd - 2)
    y1 = y0 + 1
    sx = (px - np.floor(px)).astype(np.float32)
    sy = (py - np.floor(py)).astype(np.float32)

    vals = (
        (1 - sx) * ((1 - sy) * strip[x0, y0] + sy * strip[x0, y1]) +
        sx       * ((1 - sy) * strip[x1, y0] + sy * strip[x1, y1])
    )

    vals *= side_falloff
    vals[~mask] = 0.0

    return vals.reshape(out_h, out_w).astype(np.float32)

# Interpolation of one segment
def tj(ti, pi, pj, alpha=0.5):
    return ti + max(np.linalg.norm(pj - pi) ** alpha, 1e-4)

# Interpolation of one segment
def catmull_rom_one(p0, p1, p2, p3, n=40, alpha=0.5):

    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)

    t0 = 0.0
    t1 = tj(t0, p0, p1, alpha)
    t2 = tj(t1, p1, p2, alpha)
    t3 = tj(t2, p2, p3, alpha)

    ts = np.linspace(t1, t2, n, endpoint=False, dtype=np.float32)
    pts = []

    for t in ts:
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        pts.append(C)

    return np.asarray(pts, dtype=np.float32)
# Interpolation of n segments
def sample_catmull_rom_chain(ctrl, samples_per_seg=40, alpha=0.5):

    ctrl = np.asarray(ctrl, dtype=np.float32)
    assert ctrl.shape[0] >= 2

    if ctrl.shape[0] == 2:
        # fallback: straight line
        t = np.linspace(0.0, 1.0, samples_per_seg + 1, dtype=np.float32)
        return (1.0 - t[:, np.newaxis]) * ctrl[0] + t[:, np.newaxis] * ctrl[1]

    # endpoint extrapolation
    p_before = 2.0 * ctrl[0] - ctrl[1]
    p_after  = 2.0 * ctrl[-1] - ctrl[-2]
    ext = np.vstack([p_before[np.newaxis, :], ctrl, p_after[np.newaxis, :]])

    pts = []
    for i in range(len(ctrl) - 1):
        p0, p1, p2, p3 = ext[i:i+4]
        span = catmull_rom_one(p0, p1, p2, p3, n=samples_per_seg, alpha=alpha)
        pts.append(span)

    pts.append(ctrl[-1][np.newaxis, :])
    return np.vstack(pts).astype(np.float32)
# Precompute segment info for a centerline
def build_centerline_segments(P):

    P = np.asarray(P, dtype=np.float32)
    seg = P[1:] - P[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len = np.maximum(seg_len, 1e-8)

    seg_t = seg / seg_len[:, np.newaxis]
    seg_n = np.stack([-seg_t[:, 1], seg_t[:, 0]], axis=1)

    S = np.zeros(len(P), dtype=np.float32)
    S[1:] = np.cumsum(seg_len)
    L = float(S[-1])

    return {
        "P": P,
        "seg": seg,
        "seg_len": seg_len,
        "seg_t": seg_t,
        "seg_n": seg_n,
        "S": S,
        "L": L
    }

if __name__ == "__main__":
    # strip()
    # strip_img = np.load("data/kh_curtain_strip.npz")["strip"].astype(np.float32)
    ctrl = np.array([
        [-220.0,   0.0],
        [-150.0,  10.0],
        [ -80.0,  -6.0],
        [   0.0,   8.0],
        [  90.0,  -4.0],
        [ 180.0,   9.0],
        [ 260.0,   2.0],
    ], dtype=np.float32)

    curtain_width = 4.0

    # KH strip
    strip_img = simulate(h=1024, w=96)

    # Catmull-Rom centerline
    P = sample_catmull_rom_chain(ctrl, samples_per_seg=80, alpha=0.5)

    # segment-based frame
    frame = build_centerline_segments(P)

    # better repeat length
    repeat_length = float(frame["L"])

    # raster bounds
    pad = curtain_width * 0.75
    xmin = P[:, 0].min() - pad
    xmax = P[:, 0].max() + pad
    ymin = P[:, 1].min() - pad
    ymax = P[:, 1].max() + pad

    # wrap footprint
    px_per_world = 1024.0 / frame["L"] * 2  
    out_w = int((xmax - xmin) * px_per_world)
    out_h = int((ymax - ymin) * px_per_world)
    out_w = min(out_w, 4096)
    out_h = min(out_h, 4096)

    img = rasterize_wrapped_footprint(
        strip_img, frame,
        curtain_width,
        repeat_length,
        xmin, xmax, ymin, ymax,
        out_h=out_h, out_w=out_w   
    )
    
    np.savez_compressed(
        "kh_footprint.npz",
        strip=strip_img.astype(np.float32),  
        img=img.astype(np.float32),          
        P=P.astype(np.float32),
        seg_len=frame["seg_len"].astype(np.float32),
        seg_t=frame["seg_t"].astype(np.float32),
        seg_n=frame["seg_n"].astype(np.float32),
        S=frame["S"].astype(np.float32),
        L=np.array([frame["L"]], dtype=np.float32),
        curtain_width=np.array([curtain_width], dtype=np.float32),
        repeat_length=np.array([repeat_length], dtype=np.float32),
        bounds=np.array([xmin, xmax, ymin, ymax], dtype=np.float32),
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(
        img,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="gray",
        aspect="equal"
    )
    plt.title("Wrapped aurora footprint")
    plt.xlabel("world x")
    plt.ylabel("world y")
    plt.colorbar()
    plt.tight_layout()
    plt.show()