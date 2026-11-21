#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in float vEmissive;

uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform vec3 uFogColor;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 24.0) * 0.3;
    float hemi = 0.5 + 0.5 * N.y;
    vec3 ambient = mix(vec3(0.22, 0.21, 0.20), vec3(0.45, 0.48, 0.52), hemi);
    vec3 c = vColor * (ambient + 0.8 * diff) + vec3(spec);
    c = mix(c, vColor * 1.25 + vec3(0.15), vEmissive);
    float d = length(uCamPos - vWorld);
    float fog = clamp((d - 180.0) / 500.0, 0.0, 0.6);
    FragColor = vec4(mix(c, uFogColor, fog), 1.0);
}
