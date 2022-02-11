#version 450
// Basic GLSL shader for disparity:
// Takes in the frames from both cameras and 
// Spits out a depth map

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// TODO propper kernels
layout(set = 0, binding = 0) uniform sampler2D img1;
layout(set = 0, binding = 1) uniform sampler2D img2;

layout(set = 0, binding = 2, rgba8) uniform writeonly image2D depth;

// TODO calibrate
const int baseline = 1;
const int calibration_coefficient = 1;
const int dxr = 1;
const int dyr = 1;

const int eval_latancy = 1;
const int latency = 1;

void main() {
    float x = gl_GlobalInvocationID.x; 
    float y = gl_GlobalInvocationID.y;

    // TODO Note tradeoffs
    vec3 px1 = texture(img1, vec2(x, y)).rgb;
    vec3 px2 = texture(img1, vec2(x - 1, y)).rgb;
    vec3 px3 = texture(img1, vec2(x, y - 1)).rgb;
    vec3 px4 = texture(img1, vec2(x - 1, y - 1)).rgb;
    
    vec3 hash1 = (px1 + px2 + px3 + px4) * 1000;

    int min_xr = int(x / 2); // trunk for now
    int max_xr = int((x + dxr) / 2.0);
    
    int min_yr = int(y / 2.0);
    int max_yr = int((y + dyr) / 2.0);

    vec2 best_pos = vec2(0, 0);
    int best_score = latency;
    for (int yrg = min_yr; yrg < max_yr; yrg++) {
        for (int xrg = min_xr; xrg < max_xr; xrg++) {
            vec3 px = texture(img2, vec2(xrg*2, yrg*2)).rgb * 1000;
            if (int(hash1.x) % int(px.x) + 
                int(hash1.y) % int(px.y) + 
                int(hash1.z) % int(px.z) < eval_latancy) {
                
                vec3 gpx1 = texture(img2, vec2(xrg, yrg)).rgb;
                vec3 gpx2 = texture(img2, vec2(xrg-1, yrg)).rgb;
                vec3 gpx3 = texture(img2, vec2(xrg, yrg-1)).rgb;
                vec3 gpx4 = texture(img2, vec2(xrg-1, yrg-1)).rgb;

                vec3 hash2 = px1 + px2 + px3 + px4;

                vec3 diff = (hash1 - hash2);

                int similarity = int(diff.x + diff.y + diff.z); // for now

                if (similarity < latency && best_score > similarity) {
                    best_score = similarity;
                    best_pos = vec2(xrg, yrg);
                }
            }
        }
    }
    
    vec4 to_write = vec4(baseline * calibration_coefficient / (x - best_pos.x), 0, 0, 0); // for now
    imageStore(depth, ivec2(gl_GlobalInvocationID.xy), to_write);
}