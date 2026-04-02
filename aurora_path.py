import drjit as dr
import mitsuba as mi
import numpy as np

class AuroraPath(mi.python.ad.integrators.common.RBIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        # Aurora offset
        self.aurora_x = props.get("aurora_x", 0.0)
        self.aurora_y = props.get("aurora_y", 0.0)
        self.aurora_z = props.get("aurora_z", 0.0)

        self.aurora_offset = mi.Vector3f( self.aurora_x, self.aurora_y, self.aurora_z)


        volume_path = props.get("volume_path", "aurora_volume.npz")
        # params
        self.step_size = props.get("step_size", 0.5)   
        self.scale = props.get("scale", 1.0)
        self.max_depth = props.get("max_depth", 8)
        self.direct_scale = props.get("direct_scale", 1e-4)
        data = np.load(volume_path)

        emission_rgb = data["emission_rgb"].astype(np.float32)   # (nz, ny, nx, 3)
        bbox_min = data["bbox_min"].astype(np.float32)
        bbox_max = data["bbox_max"].astype(np.float32)
        
        # texture
        self.emission = mi.Texture3f(mi.TensorXf(emission_rgb))

        # BBox  
        self.local_bbox_min = mi.Point3f(*bbox_min.tolist())
        self.local_bbox_max = mi.Point3f(*bbox_max.tolist())

        self.bbox_min = self.local_bbox_min + self.aurora_offset
        self.bbox_max = self.local_bbox_max + self.aurora_offset
        self.bbox = mi.BoundingBox3f(self.bbox_min, self.bbox_max)

        self.extent = self.local_bbox_max - self.local_bbox_min
        # russian roullete
        self.rr_depth = props.get("rr_depth", 3)

        self.ny, self.nx = data["footprint_xy"].shape
        self.nz = emission_rgb.shape[0]

        self.dx = float(data["dx"][0])
        self.dy = float(data["dy"][0])
        self.dz = float(data["dz"][0])

        # CDF PDF 2D 
        self.pdf_xy = mi.Float(data["pdf_xy"].ravel().astype(np.float32))
        self.cdf_xy = mi.Float(data["cdf_xy"].ravel().astype(np.float32))

        # CDF PDF vertical  
        self.pdf_z = mi.Float(data["pdf_z"].astype(np.float32))
        self.cdf_z = mi.Float(data["cdf_z"].astype(np.float32))

        self.nxy = self.nx * self.ny
        self.bg_color = mi.Spectrum(0.0, 0.0, 0.0)
        print("emission shape:", emission_rgb.shape)
        print("bbox_min:", self.bbox_min)
        print("bbox_max:", self.bbox_max)

    def sample_cdf_1d(self, cdf, n, u):
        lo = mi.UInt32(0)
        hi = mi.UInt32(n - 1)

        iters = int(np.ceil(np.log2(n)))
        for _ in range(iters):
            mid = (lo + hi) >> 1
            c = dr.gather(mi.Float, cdf, mid)
            go_right = u > c
            lo = dr.select(go_right, mid + 1, lo)
            hi = dr.select(go_right, hi, mid)

        return dr.minimum(lo, mi.UInt32(n - 1))
    
    # Importance sampling aurora for dl
    def sample_importance_aurora_point(self, sampler, active):
        active = mi.Bool(active)

        # sample 2d footprint 
        u_xy = sampler.next_1d(active)
        flat = self.sample_cdf_1d(self.cdf_xy, self.nxy, u_xy)

        iy = flat // self.nx
        ix = flat - iy * self.nx

        # sample vertical profile
        u_z = sampler.next_1d(active)
        iz = self.sample_cdf_1d(self.cdf_z, self.nz, u_z)

        # jitter
        jx = sampler.next_1d(active)
        jy = sampler.next_1d(active)
        jz = sampler.next_1d(active)

        x = self.bbox_min.x + (mi.Float(ix) + jx) * self.dx
        y = self.bbox_min.y + (mi.Float(iy) + jy) * self.dy
        z = self.bbox_min.z + (mi.Float(iz) + jz) * self.dz

        pL = mi.Point3f(x, y, z)

        # continuous pdf 
        pdf_xy_disc = dr.gather(mi.Float, self.pdf_xy, flat)
        pdf_z_disc  = dr.gather(mi.Float, self.pdf_z, iz)

        pdf_xy_cont = pdf_xy_disc / (self.dx * self.dy)
        pdf_z_cont  = pdf_z_disc / self.dz
        pdf = pdf_xy_cont * pdf_z_cont

        return pL, pdf
    def world_to_local(self, p):
        p_local = p - self.aurora_offset
        return (p_local - self.local_bbox_min) / self.extent

    def traverse(self, cb):
        cb.put("emission", self.emission.tensor(), mi.ParamFlags.NonDifferentiable)
    def parameters_changed(self, keys):
        self.emission.set_tensor(self.emission.tensor())
    
    @dr.syntax
    def raymarch_segment(self, ray, t_end, sampler, active):
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        hit, mint, maxt = self.bbox.ray_intersect(ray)
        active &= hit

        t0 = dr.maximum(mi.Float(0.0), mint)
        t1 = dr.minimum(t_end, maxt)
        active &= (t1 > t0)

        L = mi.Spectrum(0.0)
        step = mi.Float(self.step_size)

        t = t0 + sampler.next_1d(active) * step

        while dr.hint(active, mode="symbolic"):
            p = ray(t)
            local = self.world_to_local(p)

            inside = ((local.x >= 0.0) & (local.x <= 1.0) & (local.y >= 0.0) & (local.y <= 1.0) & (local.z >= 0.0) & (local.z <= 1.0))

            eval_active = active & inside & (t < t1)

            rgb = self.emission.eval(local)
            L[eval_active] += mi.Spectrum(rgb) * step * self.scale

            t += step
            active &= (t < t1)

        return L
    # Direct lighting per surface hit
    def sample_direct_aurora(self, si, scene, sampler, active):
        active = mi.Bool(active)

        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        n_light = 16
        Ld = mi.Spectrum(0.0)

        for _ in range(n_light):
            pL, pdf = self.sample_importance_aurora_point(sampler, active)

            # evaluate aurora emission at sampled point
            local = self.world_to_local(pL)
            Le = mi.Spectrum(self.emission.eval(local)) * self.scale

            sample_active = active & (pdf > 0.0) & (dr.max(Le) > 0.0)

            d = pL - si.p
            dist2 = dr.maximum(dr.dot(d, d), 1e-4)
            wi_world = d * dr.rsqrt(dist2)

            shadow_ray = si.spawn_ray_to(pL)
            visible = sample_active & ~scene.ray_test(shadow_ray, sample_active)

            wi_local = si.to_local(wi_world)
            f = bsdf.eval(ctx, si, wi_local, visible)

            geom = dr.rcp(dist2)

            contrib = self.direct_scale * f * Le * geom / (pdf * n_light)
            Ld += dr.select(visible, contrib, mi.Spectrum(0.0))

        return Ld
    # environment lighting for miss rays
    def eval_environment(self, scene, ray, active):
        env = scene.environment()
        if env is None:
            return mi.Spectrum(0.0)

        si_env = mi.SurfaceInteraction3f()
        si_env.wi = -ray.d
        si_env.time = ray.time
        si_env.p = ray.o

        return mi.Spectrum(env.eval(si_env, active))
    
    # Path tracing, from Mitsuba NERF guide
    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        L = mi.Spectrum(0.0)
        beta = mi.Spectrum(1.0)
        depth = mi.UInt32(0)

        ctx = mi.BSDFContext()
        has_env = scene.environment() is not None

        while dr.hint(active, mode="symbolic", max_iterations=-1):
            si = scene.ray_intersect(ray, active)
            valid_si = si.is_valid()
            t_surface = dr.select(valid_si, si.t, mi.Float(dr.inf))

            # aurora along current segment
            L += beta * self.raymarch_segment(ray, t_surface, sampler, active)

            # environment on miss rays
            miss = active & ~valid_si
            if has_env:
                env_L = self.eval_environment(scene, ray, miss)
                L += dr.select(miss, beta * env_L, mi.Spectrum(0.0))

            # direct lighting on hit surfaces
            surface_active = active & valid_si
            L += beta * self.sample_direct_aurora(si, scene, sampler, surface_active)

            active &= valid_si
            active &= (depth < self.max_depth)

            bsdf = si.bsdf(ray)
            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

            beta[active] *= bsdf_weight
            
            # Russian roulette
            rr_active = active & (depth >= self.rr_depth)
            q = dr.clamp(dr.max(beta), 0.05, 0.95)

            survive = sampler.next_1d(rr_active) < q
            survive_rr = rr_active & survive

            beta[survive_rr] *= dr.rcp(q)
            active &= (~rr_active) | survive
        
            ray[active] = si.spawn_ray(si.to_world(bs.wo))
            depth[active] += 1

        return L if primal else δL, mi.Bool(True), [], L
mi.register_integrator("aurora_path", lambda props: AuroraPath(props))