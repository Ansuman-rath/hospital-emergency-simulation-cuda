#version 330 core
// Instanced patients / staff. One draw call renders every patient; per-instance
// data (x, z, packed state|priority<<8|flags<<16) comes straight from the
// simulation (optionally written by CUDA through CUDA-OpenGL interop).
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in float aPart;      // 0 body, 1 head, 2 status marker
layout(location = 4) in vec2 iPos;
layout(location = 5) in uint iPacked;

uniform mat4 uViewProj;
uniform float uTime;
uniform int uSelected;          // selected patient id (-1 none)
uniform int uMode;              // 0 = patients, 1 = staff
uniform int uHidePriorityMask;  // bit p set -> hide priority p
uniform vec3 uPriorityColor[4];
uniform vec3 uStateColor[14];

out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
out float vEmissive;

const int ST_NOT_ARRIVED = 0;
const int ST_DISCHARGED = 13;
const int FLAG_MOVING = 1;
const int FLAG_TRIAGED = 2;

void main() {
    int state = int(iPacked & 0xFFu);
    int prio = int((iPacked >> 8) & 0xFFu);
    int flags = int((iPacked >> 16) & 0xFFu);

    vec3 p = aPos;
    vec3 n = aNormal;
    vec3 col;
    float emissive = 0.0;
    float scale = 1.0;

    if (uMode == 0) {
        bool hidden = state == ST_NOT_ARRIVED || state == ST_DISCHARGED || ((uHidePriorityMask >> prio) & 1) == 1;
        if (hidden) {  // degenerate -> clipped
            gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
            vWorld = vec3(0.0); vNormal = vec3(0.0, 1.0, 0.0); vColor = vec3(0.0); vEmissive = 0.0;
            return;
        }
        bool triaged = (flags & FLAG_TRIAGED) != 0;
        bool moving = (flags & FLAG_MOVING) != 0;
        // lying in a bed: treatment (7), boarding (8, 9), ICU (10), ward (11)
        bool lying = !moving && state >= 7 && state <= 11;
        bool waiting = state == 2 || state == 4 || state == 6 || state == 8 || state == 9;
        if (aPart < 0.5) {
            col = triaged ? uPriorityColor[prio] : vec3(0.85, 0.86, 0.90);
        } else if (aPart < 1.5) {
            col = vec3(0.96, 0.80, 0.66);
        } else {
            col = uStateColor[state];
            if (waiting) emissive = 0.5 + 0.5 * sin(uTime * 5.0);
            p.y += 0.05 * sin(uTime * 3.0 + float(gl_InstanceID));
        }
        if (lying) {  // rotate the figure to lie along +z on the bed
            p = vec3(p.x, 0.75 + p.z, p.y - 0.55);
            n = vec3(n.x, n.z, n.y);
        }
        if (gl_InstanceID == uSelected) {
            scale = 1.6;
            emissive = max(emissive, 0.6);
        }
    } else {
        int kind = int(iPacked & 0xFFu);  // 0 busy doctor, 1 idle doctor, 2 busy nurse, 3 idle nurse
        if (aPart < 0.5) col = kind < 2 ? vec3(0.92, 0.95, 1.0) : vec3(0.40, 0.75, 0.60);
        else if (aPart < 1.5) col = vec3(0.90, 0.74, 0.60);
        else col = (kind == 0 || kind == 2) ? vec3(0.10, 0.55, 1.0) : vec3(0.55, 0.55, 0.60);
        scale = 1.05;
    }

    vec3 world = vec3(iPos.x, 0.0, iPos.y) + p * scale;
    vWorld = world;
    vNormal = n;
    vColor = col;
    vEmissive = emissive;
    gl_Position = uViewProj * vec4(world, 1.0);
}
