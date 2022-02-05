#version 450
// Basic GLSL shader for disparity:
// Takes in the frames from both cameras and 
// Spits out a depth map

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// TODO propper kernels
layout(set = 0, binding = 0, rgba8) uniform readonly sampler2D img1;
layout(set = 0, binding = 1, rgba8) uniform readonly sampler2D img2;

layout(set = 0, binding = 2, rgba8) uniform writeonly image2D depth;

// TODO calibrate
const int baseline = ();
const int calibration_coefficient = ();
const int dxr = ();
const int dyr = ();

const int eval_latancy = ();
const int latency = ();

void main() {
    int x = gl_FragCoord.x; int y = gl_FragCoord.y;

    // TODO Note tradeoffs
    vec4 px1 = img1(x, y);
    vec4 px2 = img1(x - 1, y);
    vec4 px3 = img1(x, y - 1);
    vec4 px4 = img1(x - 1, y - 1);
    
    vec4 hash1 = px1 + px2 + px3 + px4;

    int min_xr = x / 2;
    int max_xr = (x + dxr) / 2;
    
    int min_yr = y / 2;
    int max_yr = (y + dyr) / 2;

    vec2 best_pos = vec2(0, 0);
    int best_score = latency;
    for (int yrg = min_yr; yrg < max_yr; yrg++) {
        for (int xrg = min_xr; xrg < max_xr; xrg++) {
            if ((hash % img2(xrg*2, yrg*2)) < eval_latancy) {
                vec4 gpx1 = img2(xrg, yrg);
                vec4 gpx2 = img2(xrg-1, yrg);
                vec4 gpx3 = img2(xrg, yrg-1);
                vec4 gpx4 = img2(xrg-1, yrg-1);

                vec4 hash2 = px1 + px2 + px3 + px4;

                int similarity = hash1 - hash2; // for now

                if (similarity < latency && best_score > similarity) {
                    best_score = similarity;
                    best = (xrg, yrg)
                }
            }
        }
    }
    
    vec4 to_write = baseline * calibration_coefficient / (x - best_pos.x);
    imageStore(depth, ivec2(gl_GlobalInvocationID.xy), to_write);
}