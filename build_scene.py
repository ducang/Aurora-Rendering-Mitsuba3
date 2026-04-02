import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import aurora_path

scene = mi.load_dict({
    "type": "scene",

    "integrator": {
        "type": "aurora_path",
        "volume_path": "aurora_volume.npz",
        "step_size": 0.1,
        "scale": 1.5,
        "max_depth": 10,
        "direct_scale": 1e-5,
        "aurora_x": -5.0,
        "aurora_y": 50.0,
        "aurora_z": -40.0
    },

    "sensor": {
        "type": "perspective",
        "fov": 50,
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[0, -190, 40],
            target=[0, 20, 20],
            up=[0, 0, 1]
        ),
        "sampler": {
            "type": "independent",
            "sample_count": 128
        },
        "film": {
            "type": "hdrfilm",
            "width": 1920,
            "height": 1080,
            "rfilter": {"type": "gaussian"}
        }
    },
    "sky": {
        "type": "constant",
        "radiance": {
            "type": "rgb",
            "value": [0.001, 0.001, 0.002]
        }
        # "type": "envmap",
        # "filename": "bg1.png",
        # "scale": 1,
    },
    "mountain": {
        "type": "obj",
        "filename": "assets/mesh/Mountain.obj",
        "to_world": mi.ScalarTransform4f()
                        .translate([0, -13, 4])
                        .rotate([0, 0, 1], 180)
                        .rotate([1, 0, 0], 90)
                        .scale(8.0),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "filename": "assets/textures/terrain_color_01.png",
                "type": "bitmap"
            }
        }
    },
    "lake": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f()
            .translate([0, 30, 0])
            .scale([200, 120, 1]),
        "bsdf": {
            "type": "roughdielectric",
            "alpha": 1,
            "distribution": "ggx",
            "int_ior": "water",
            "ext_ior": "air"
        }
    },

})

img = mi.render(scene, spp=2048)
mi.util.write_bitmap("test.png", img ** (1.0 / 2.2))
print("done")