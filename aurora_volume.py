import numpy as np

# MSIS-E-90 atmosphere approximation (80–300 km)
# Each layer: (z_base_km, density_g_cm3, scale_height_km)
MSIS_LAYERS = np.array([
    ( 80.0, 1.85e-8,   6.0),
    ( 90.0, 3.20e-9,   6.5),
    (100.0, 5.60e-10,  7.5),
    (110.0, 9.00e-11,  9.5),
    (120.0, 2.20e-11, 14.0),
    (140.0, 3.80e-12, 22.0),
    (160.0, 1.05e-12, 30.0),
    (180.0, 3.80e-13, 38.0),
    (200.0, 1.60e-13, 44.0),
    (240.0, 3.60e-14, 55.0),
    (300.0, 5.00e-15, 65.0),
], dtype=np.float64)


def msis_density(z_km):
    z = np.asarray(z_km, dtype=np.float64)
    D = np.zeros_like(z)
    for i in range(len(MSIS_LAYERS) - 1):
        z0, D0, H = MSIS_LAYERS[i]
        z1 = MSIS_LAYERS[i + 1, 0]
        mask = (z >= z0) & (z < z1)
        D[mask] = D0 * np.exp(-(z[mask] - z0) / H)
    z0, D0, H = MSIS_LAYERS[-1]
    mask = z >= z0
    D[mask] = D0 * np.exp(-(z[mask] - z0) / H)
    return D


def msis_column_mass(z_km, dz=0.5):

    z_top = 600.0
    z_grid = np.arange(0.0, z_top + dz, dz, dtype=np.float64)  
    D_grid = msis_density(z_grid)                               

    cum = np.cumsum(D_grid[::-1])[::-1] * dz * 1e5             
    z_scalar = np.atleast_1d(np.asarray(z_km, dtype=np.float64))
    Mz = np.interp(z_scalar, z_grid, cum)
    return Mz if np.ndim(z_km) > 0 else float(Mz[0])


# Lazarev deposition model 
def lazarev_deposition(z_km, E_keV):

    z = np.atleast_1d(np.asarray(z_km,  dtype=np.float64))
    Dz = msis_density(z)                    
    Mz = msis_column_mass(z)                

    ME = 4.6e-6 * (E_keV ** 1.65)           
    r = Mz / ME                            

    L = (4.2 * r * np.exp(-r**2 - r) + 0.48 * np.exp(-17.4 * np.maximum(r, 0.0) ** 1.37))

    Az = L * E_keV * (Dz / ME)
    return Az.astype(np.float32)


def lazarev_deposition_summed(z_km, E_min_keV =  1.0, E_max_keV = 20.0,  n_energies =  8):
    energies = np.geomspace(E_min_keV, E_max_keV, n_energies)
    Az_sum   = np.zeros(len(np.atleast_1d(z_km)), dtype=np.float64)
    for E in energies:
        Az_sum += lazarev_deposition(z_km, E)
    return (Az_sum / max(Az_sum.max(), 1e-30)).astype(np.float32)

def project_to_centerline_batch(XY, frame):
    P       = frame["P"]        # (M, 2)
    seg_len = frame["seg_len"]  # (M-1,)
    seg_t   = frame["seg_t"]    # (M-1, 2)
    seg_n   = frame["seg_n"]    # (M-1, 2)
    S       = frame["S"]        # (M,)

    rel   = XY[:, np.newaxis, :] - P[np.newaxis, :-1, :]                     # (N, M-1, 2)
    a     = np.einsum('nij,ij->ni', rel, seg_t)                  # (N, M-1)
    tau   = np.clip(a / seg_len[np.newaxis, :], 0.0, 1.0)              # (N, M-1)

    Q     = P[np.newaxis, :-1, :] + tau[:, :, np.newaxis] * seg_len[np.newaxis, :, np.newaxis] * seg_t[np.newaxis, :, :]
    delta = XY[:, np.newaxis, :] - Q                                   # (N, M-1, 2)
    dist2 = np.sum(delta * delta, axis=2)                        # (N, M-1)

    best_i   = np.argmin(dist2, axis=1)
    idx      = np.arange(XY.shape[0])
    best_tau = tau[idx, best_i]

    s_world  = S[best_i] + best_tau * seg_len[best_i]
    d_world  = np.einsum('ni,ni->n', delta[idx, best_i], seg_n[best_i])

    return s_world.astype(np.float32), d_world.astype(np.float32)

# Sample the footprint at given XY points
def sample_footprint(XY, frame, strip, curtain_width, repeat_length):

    s_world, d_world = project_to_centerline_batch(XY, frame)

    mask = np.abs(d_world) <= 0.5 * curtain_width

    u = (s_world / repeat_length) % 1.0
    v = np.clip(d_world / curtain_width + 0.5, 0.0, 1.0)

    Hs, Hd = strip.shape
    px = u * Hs
    py = v * (Hd - 1)

    x0 = np.floor(px).astype(np.int32) % Hs
    x1 = (x0 + 1) % Hs
    y0 = np.clip(np.floor(py).astype(np.int32), 0, Hd - 2)
    y1 = y0 + 1

    sx = (px - np.floor(px)).astype(np.float32)
    sy = (py - np.floor(py)).astype(np.float32)

    vals = (
        (1.0 - sx) * ((1.0 - sy) * strip[x0, y0] + sy * strip[x0, y1]) +
        sx         * ((1.0 - sy) * strip[x1, y0] + sy * strip[x1, y1])
    ).astype(np.float32)

    vals[~mask] = 0.0
    return vals

THRESHOLD = 0.05
GAUSS_K   = np.sqrt(-2.0 * np.log(THRESHOLD))


def build_vertical_lut(zmin=80.0, zmax=300.0, nz=256,
    #Lazarev electron energy range 
    E_min_keV  =  1.0,   # raise to push green band lower
    E_max_keV  = 20.0,   # lower to raise and soften peak
    n_energies =  8,     # more = smoother curve
):
    # Color param to create color LUT, following Figure 13a in Baranoski et al.

    # Green
    GREEN_START=100.0
    GREEN_PEAK=138.0
    GREEN_END=260.0
    GREEN_AMP=1.00,

    # Blue
    BLUE_START=100.0
    BLUE_PEAK=138.0
    BLUE_END=249.0
    BLUE_AMP=0.50

    # Red 
    RED_START=165.0
    RED_PEAK=270.0
    RED_TOP_VAL=0.15
    RED_AMP=0.30

    z = np.linspace(zmin, zmax, nz, dtype=np.float32)

    def gauss_curve(z, start, peak, end, amp):
        sl = (peak - start) / GAUSS_K
        sh = (end  - peak)  / GAUSS_K
        s  = np.where(z <= peak, sl, sh)
        return amp * np.exp(-0.5 * ((z - peak) / s) ** 2)

    green_c = gauss_curve(z, GREEN_START, GREEN_PEAK, GREEN_END, GREEN_AMP)
    blue_c = gauss_curve(z, BLUE_START,  BLUE_PEAK,  BLUE_END,  BLUE_AMP)

    sl = (RED_PEAK - RED_START) / GAUSS_K
    ratio = np.clip(RED_TOP_VAL / RED_AMP, 1e-8, 1.0 - 1e-8)
    sh = (zmax - RED_PEAK) / np.sqrt(-2.0 * np.log(ratio))
    s = np.where(z <= RED_PEAK, sl, sh)
    red_c = (RED_AMP * np.exp(-0.5 * ((z - RED_PEAK) / s) ** 2)).astype(np.float32)

    C_rgb = np.stack([red_c, green_c, blue_c], axis=1).astype(np.float32)
    C_rgb /= max(float(C_rgb.max()), 1e-8)


    A = lazarev_deposition_summed(z.astype(np.float64), E_min_keV=E_min_keV, E_max_keV=E_max_keV, n_energies=n_energies)
    A /= max(float(A.max()), 1e-8)

    return z, A, C_rgb
def sample_vertical_lut_batch(z_query, z_lut, A_lut, C_rgb_lut):

    A = np.interp(z_query, z_lut, A_lut).astype(np.float32)
    Cr = np.interp(z_query, z_lut, C_rgb_lut[:, 0]).astype(np.float32)
    Cg = np.interp(z_query, z_lut, C_rgb_lut[:, 1]).astype(np.float32)
    Cb = np.interp(z_query, z_lut, C_rgb_lut[:, 2]).astype(np.float32)
    C = np.stack([Cr, Cg, Cb], axis=1).astype(np.float32)
    return A, C


def choose_xy_resolution(bounds, max_xy=384):
    xmin, xmax, ymin, ymax = bounds
    sx = float(xmax - xmin)
    sy = float(ymax - ymin)
    if sx >= sy:
        nx = max_xy
        ny = max(2, int(round(max_xy * sy / max(sx, 1e-8))))
    else:
        ny = max_xy
        nx = max(2, int(round(max_xy * sx / max(sy, 1e-8))))
    return nx, ny

# Perlin noise helpers
def perlin_fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def perlin_grad1d(h, x):
    return np.where(h & 1, x, -x)

# 1D and 2D Perlin noise implementations 
def perlin_1d(x, seed=0):

    rng  = np.random.default_rng(seed)
    N = 256
    perm = rng.permutation(N).astype(np.int32)
    perm = np.concatenate([perm, perm])  

    xi = np.floor(x).astype(np.int32) & 255
    xf = x - np.floor(x)
    u = perlin_fade(xf)

    g0 = perlin_grad1d(perm[xi], xf)
    g1 = perlin_grad1d(perm[xi + 1], xf - 1.0)
    return g0 + u * (g1 - g0)         

def perlin_2d(x, y, seed=0):

    rng = np.random.default_rng(seed)
    N = 256
    perm = rng.permutation(N).astype(np.int32)
    perm = np.concatenate([perm, perm])

    def grad2(h, dx, dy):
        h = h & 3
        u = np.where(h < 2, dx, dy)
        v = np.where(h < 2, dy, dx)
        return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)

    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    u  = perlin_fade(xf)
    v  = perlin_fade(yf)

    aa = perm[perm[xi]+ yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x0 = grad2(aa, xf, yf) + u * (grad2(ba, xf - 1, yf) - grad2(aa, xf, yf))
    x1 = grad2(ab, xf, yf - 1) + u * (grad2(bb, xf - 1, yf - 1) - grad2(ab, xf, yf - 1))
    return x0 + v * (x1 - x0)

# Tao et al. style noise for aurora visual 
def build_tao_noises(s_map, XX, YY, L):

    # noise 1: high freq 1D perturbs base altitude 
    N1_AMP = 0.8  # how much the lower border jiggles
    N1_FREQ = 2.5 # cycles per curtain length 
    N1_SEED = 42
    # noise 2: low freq 1D perturbs color LUT z
    N2_AMP = 18.0 # how much color shifts along strip
    N2_FREQ = 1.5 # cycles per curtain 
    N2_SEED = 137
    # noise 3: medium freq 2D perturbs footprint density 
    N3_AMP = 0.30 #density variation amplitude
    N3_FREQ = 0.025 # cycles per world unit 
    N3_SEED = 901

    # noise 1: 1D along arclength, high frequency 
    s01= s_map / max(float(L), 1e-8)         
    n1 = perlin_1d(s01 * N1_FREQ * 2 * np.pi, seed=N1_SEED)
    noise1 = (N1_AMP * n1).astype(np.float32)   

    # noise 2: 1D along arclength, low frequency 
    n2 = perlin_1d(s01 * N2_FREQ * 2 * np.pi, seed=N2_SEED)
    noise2 = (N2_AMP * n2).astype(np.float32)

    # noise 3: 2D in world XY, medium frequency 
    n3 = perlin_2d(XX * N3_FREQ, YY * N3_FREQ, seed=N3_SEED)
    noise3= np.clip(1.0 + N3_AMP * n3, 0.0, 2.0).astype(np.float32)

    return noise1, noise2, noise3

def probability_tables(footprint_xy, A_lut, xy_power=1.5, z_power=1.0):
    # 2D footprint weights
    w_xy = np.maximum(footprint_xy.astype(np.float64), 0.0) ** xy_power + 1e-12
    pdf_xy = w_xy / w_xy.sum()
    cdf_xy = np.cumsum(pdf_xy.ravel())
    cdf_xy[-1] = 1.0

    # 1D vertical weights
    w_z = np.maximum(A_lut.astype(np.float64), 0.0) ** z_power + 1e-12
    pdf_z = w_z / w_z.sum()
    cdf_z = np.cumsum(pdf_z)
    cdf_z[-1] = 1.0

    return (pdf_xy.astype(np.float32),cdf_xy.astype(np.float32),pdf_z.astype(np.float32),cdf_z.astype(np.float32),)
def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def bake_volume(
    footprint_npz="kh_footprint.npz",
    out_npz="aurora_volume.npz",

    # actual visible curtain height in the scene
    geom_zmin=80.0,
    geom_zmax=170.0,

    # full lookup height for intensity+color
    lut_zmin=80.0,
    lut_zmax=300.0,

    nz=128,
    max_xy=256,

    # Lazarev energy range
    E_min_keV=1.0,
    E_max_keV=20.0,
    n_energies=8,

    # actual curtain column height in world space
    base_geom_height= None,
):
    data = np.load(footprint_npz)

    strip = data["strip"].astype(np.float32)
    P = data["P"].astype(np.float32)
    seg_len = data["seg_len"].astype(np.float32)
    seg_t  = data["seg_t"].astype(np.float32)
    seg_n = data["seg_n"].astype(np.float32)
    S = data["S"].astype(np.float32)
    L = float(data["L"][0])
    curtain_width = float(data["curtain_width"][0])
    repeat_length = float(data["repeat_length"][0])
    xmin, xmax, ymin, ymax = data["bounds"].astype(np.float32)

    frame = {"P": P, "seg_len": seg_len, "seg_t": seg_t, "seg_n": seg_n, "S": S, "L": L}

    if base_geom_height is None:
        base_geom_height = geom_zmax - geom_zmin

    # Build the vertical LUT on the FULL lookup range
    z_lut, A_lut, C_rgb_lut = build_vertical_lut(
        zmin=lut_zmin,
        zmax=lut_zmax,
        nz=max(512, 2 * nz),
        E_min_keV=E_min_keV,
        E_max_keV=E_max_keV,
        n_energies=n_energies,
    )

    # Build the actual scene volume on visible range
    nx, ny = choose_xy_resolution((xmin, xmax, ymin, ymax), max_xy=max_xy)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    zs = np.linspace(geom_zmin, geom_zmax, nz, dtype=np.float32)

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # sample footrprint 
    footprint_xy = sample_footprint(XY, frame, strip, curtain_width, repeat_length).reshape(ny, nx).astype(np.float32)

    # Along-curtain coordinate
    s_world, _ = project_to_centerline_batch(XY, frame)
    s_map = s_world.reshape(ny, nx).astype(np.float32)
    s01 = s_map / max(L, 1e-8)

    # Verical shift of curtain
    z_shift_xy = (2.0 * np.sin(2.0 * np.pi * 0.45 * s01 + 0.3) + 1.0 * np.sin(2.0 * np.pi * 1.0  * s01 + 1.1)).astype(np.float32)

    height_scale_xy = (1.0 + 0.08 * np.sin(2.0 * np.pi * 0.6 * s01 + 0.8) + 0.04 * np.sin(2.0 * np.pi * 1.2 * s01 + 2.0)).astype(np.float32)
    height_scale_xy = np.clip(height_scale_xy, 0.90, 1.10)

    # Tao et al. noise step
    noise1, noise2, noise3 = build_tao_noises(s_map, XX, YY, L)

    # apply noise1 to lower border
    z_shift_xy += noise1

    # apply noise3 to  density
    footprint_xy *= noise3
    footprint_xy = np.clip(footprint_xy, 0.0, 1.0)

    # Build pdf and cdf tables
    pdf_xy, cdf_xy, pdf_z, cdf_z = probability_tables(footprint_xy, A_lut, xy_power=1.5, z_power=1.0)

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (geom_zmax - geom_zmin) / nz

    # compute bottom and top border of the curtain at each XY, apply shift and height scale
    z_bot_xy = geom_zmin + z_shift_xy
    z_top_xy = z_bot_xy + base_geom_height * height_scale_xy

    z_bot_xy = np.minimum(z_bot_xy, geom_zmax - 1.0)
    z_top_xy = np.maximum(z_top_xy, z_bot_xy + 1.0)

    # Precompute the emission RGB 
    emission_rgb = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    edge_km = 4.0  # softness of top/bottom edge 

    for iz, z in enumerate(zs):
        # normalized height inside each local curtain column
        t = (z - z_bot_xy) / np.maximum(z_top_xy - z_bot_xy, 1e-4)
        t = np.clip(t, 0.0, 1.0).astype(np.float32)

        z_lookup = lut_zmin + t * (lut_zmax - lut_zmin)

        z_color = np.clip(z_lookup + noise2, lut_zmin, lut_zmax).astype(np.float32)
        phase = 2.0 * np.pi * (0.015 * XX + 0.011 * YY + 0.020 * z)
        lookup_jitter = 3.0 * np.sin(phase).astype(np.float32)   

        z_lookup_j = np.clip(z_lookup + lookup_jitter, lut_zmin, lut_zmax).astype(np.float32)
        z_color = np.clip(z_lookup_j + noise2, lut_zmin, lut_zmax).astype(np.float32)

        A_xy, _ = sample_vertical_lut_batch(z_lookup_j.ravel(), z_lut, A_lut, C_rgb_lut)
        _, C_xy = sample_vertical_lut_batch(z_color.ravel(), z_lut, A_lut, C_rgb_lut)

        A_xy = A_xy.reshape(ny, nx)
        C_xy = C_xy.reshape(ny, nx, 3)
        w_bot = smoothstep01((z - z_bot_xy) / edge_km)
        w_top = smoothstep01((z_top_xy - z) / edge_km)
        inside_w = w_bot * w_top
        emission_rgb[iz] = (
            footprint_xy[:, :, np.newaxis]
            * inside_w[:, :, np.newaxis].astype(np.float32)
            * (A_xy[:, :, np.newaxis] * C_xy)
        )
    bbox_min = np.array([xmin, ymin, geom_zmin], dtype=np.float32)
    bbox_max = np.array([xmax, ymax, geom_zmax], dtype=np.float32)
    
    # save everything just in case
    np.savez_compressed(
        out_npz,
        emission_rgb=emission_rgb,
        footprint_xy=footprint_xy,
        xs=xs,
        ys=ys,
        zs=zs,

        z_lut=z_lut,
        A_lut=A_lut,
        C_rgb_lut=C_rgb_lut,

        pdf_xy=pdf_xy,
        cdf_xy=cdf_xy,
        pdf_z=pdf_z,
        cdf_z=cdf_z,

        dx=np.array([dx], dtype=np.float32),
        dy=np.array([dy], dtype=np.float32),
        dz=np.array([dz], dtype=np.float32),

        bbox_min=bbox_min,
        bbox_max=bbox_max,

        curtain_width=np.array([curtain_width], dtype=np.float32),
        repeat_length=np.array([repeat_length], dtype=np.float32),

        geom_zmin=np.array([geom_zmin], dtype=np.float32),
        geom_zmax=np.array([geom_zmax], dtype=np.float32),
        lut_zmin=np.array([lut_zmin], dtype=np.float32),
        lut_zmax=np.array([lut_zmax], dtype=np.float32),
        base_geom_height=np.array([base_geom_height], dtype=np.float32),
    )

    print("saved:", out_npz)
    print("emission_rgb shape:", emission_rgb.shape)
    print("bbox:", bbox_min, "→", bbox_max)

if __name__ == "__main__":
    bake_volume(
        footprint_npz="kh_footprint.npz",
        out_npz="aurora_volume.npz",
        geom_zmin=60.0,
        geom_zmax=170.0,   
        lut_zmin=50.0,
        lut_zmax=300.0,   
        nz=512,
        max_xy=384,
        E_min_keV=1.0,
        E_max_keV=20.0,
        n_energies=8,
        base_geom_height=75.0,
    )