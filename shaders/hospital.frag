#version 330 core
// Blinn-Phong: one directional "sun" light + hemisphere ambient + distance fog.
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;

uniform vec3 uCamPos;
uniform vec3 uLightDir;    // direction TO the light, normalised
uniform float uEmissive;   // 0..1 extra self-illumination (highlights)
uniform vec3 uFogColor;

out vec4 FragColor;

vec3 shade(vec3 base, vec3 n, vec3 world) {
    vec3 N = normalize(n);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - world);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.25;
    float hemi = 0.5 + 0.5 * N.y;
    vec3 ambient = mix(vec3(0.20, 0.19, 0.18), vec3(0.42, 0.45, 0.50), hemi);
    vec3 c = base * (ambient + 0.75 * diff) + vec3(spec);
    c = mix(c, base * 1.3, uEmissive);
    float d = length(uCamPos - world);
    float fog = clamp((d - 180.0) / 500.0, 0.0, 0.6);
    return mix(c, uFogColor, fog);
}

void main() {
    FragColor = vec4(shade(vColor, vNormal, vWorld), 1.0);
}
