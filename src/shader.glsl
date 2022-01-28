#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform image2D img;

void main() {
    // TODO
    vec4 to_write;
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
