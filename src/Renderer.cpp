#include "Renderer.h"

#include <cmath>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <sstream>

namespace hs {

// ----------------------------------------------------------------- geometry ----
void MeshBuilder::box(const glm::vec3& c, const glm::vec3& s, const glm::vec3& col, float part) {
    const glm::vec3 h = s * 0.5f;
    const glm::vec3 n[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int f = 0; f < 6; ++f) {
        const glm::vec3 N = n[f];
        const glm::vec3 U = std::fabs(N.y) > 0.5f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        const glm::vec3 V = glm::cross(N, U);
        const uint32_t base = (uint32_t)v.size();
        const float su[4] = {-1, 1, 1, -1}, sv[4] = {-1, -1, 1, 1};
        for (int k = 0; k < 4; ++k) v.push_back({c + (N + U * su[k] + V * sv[k]) * h, N, col, part});
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
}

void MeshBuilder::cylinder(const glm::vec3& b, float r, float hgt, int seg, const glm::vec3& col, float part) {
    const uint32_t base = (uint32_t)v.size();
    for (int i = 0; i <= seg; ++i) {
        const float a = 6.2831853f * i / seg;
        const glm::vec3 n(std::cos(a), 0, std::sin(a));
        v.push_back({b + n * r, n, col, part});
        v.push_back({b + n * r + glm::vec3(0, hgt, 0), n, col, part});
    }
    for (int i = 0; i < seg; ++i) {
        const uint32_t k = base + i * 2;
        idx.insert(idx.end(), {k, k + 1, k + 3, k, k + 3, k + 2});
    }
    const uint32_t top = (uint32_t)v.size();
    v.push_back({b + glm::vec3(0, hgt, 0), {0, 1, 0}, col, part});
    for (int i = 0; i <= seg; ++i) {
        const float a = 6.2831853f * i / seg;
        v.push_back({b + glm::vec3(std::cos(a) * r, hgt, std::sin(a) * r), {0, 1, 0}, col, part});
    }
    for (int i = 0; i < seg; ++i) idx.insert(idx.end(), {top, top + 1 + i, top + 2 + i});
}

void MeshBuilder::sphere(const glm::vec3& c, float r, int seg, int rings, const glm::vec3& col, float part) {
    const uint32_t base = (uint32_t)v.size();
    for (int j = 0; j <= rings; ++j) {
        const float t = 3.14159265f * j / rings;
        for (int i = 0; i <= seg; ++i) {
            const float a = 6.2831853f * i / seg;
            const glm::vec3 n(std::sin(t) * std::cos(a), std::cos(t), std::sin(t) * std::sin(a));
            v.push_back({c + n * r, n, col, part});
        }
    }
    for (int j = 0; j < rings; ++j)
        for (int i = 0; i < seg; ++i) {
            const uint32_t a = base + j * (seg + 1) + i, b2 = a + seg + 1;
            idx.insert(idx.end(), {a, b2, a + 1, a + 1, b2, b2 + 1});
        }
}

void GpuMesh::upload(const MeshBuilder& m, bool instanced, GLuint instanceVbo) {
    destroy();
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m.v.size() * sizeof(Vertex), m.v.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.idx.size() * 4, m.idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, part));
    if (instanced) {
        // RenderInstance and StaffInstance share the layout {float x, float z, uint32 packed}
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(RenderInstance), (void*)0);
        glVertexAttribDivisor(4, 1);
        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(5, 1, GL_UNSIGNED_INT, sizeof(RenderInstance), (void*)8);
        glVertexAttribDivisor(5, 1);
    }
    glBindVertexArray(0);
    count = (GLsizei)m.idx.size();
}

void GpuMesh::destroy() {
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    vao = vbo = ebo = 0;
    count = 0;
}

// ------------------------------------------------------------------ shaders ----
static bool readFile(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static GLuint compileStage(GLenum type, const std::string& src, std::string* err) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        if (err) *err += log;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint Renderer::compile(const std::string& vsPath, const std::string& fsPath, std::string* err) {
    std::string vs, fs;
    if (!readFile(vsPath, vs) || !readFile(fsPath, fs)) {
        if (err) *err = "cannot read shader " + vsPath + " / " + fsPath;
        return 0;
    }
    GLuint v = compileStage(GL_VERTEX_SHADER, vs, err), f = compileStage(GL_FRAGMENT_SHADER, fs, err);
    if (!v || !f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        if (err) *err += log;
        return 0;
    }
    return p;
}

bool Renderer::init(const std::string& dir, std::string* error) {
    sceneProg_ = compile(dir + "/hospital.vert", dir + "/hospital.frag", error);
    patientProg_ = compile(dir + "/patient.vert", dir + "/patient.frag", error);
    if (!sceneProg_ || !patientProg_) return false;
    glGenBuffers(1, &patientVbo_);
    glGenBuffers(1, &staffVbo_);
    reserveInstances(1024);
    glBindBuffer(GL_ARRAY_BUFFER, staffVbo_);
    staffCapacity_ = 256;
    glBufferData(GL_ARRAY_BUFFER, staffCapacity_ * sizeof(StaffInstance), nullptr, GL_DYNAMIC_DRAW);

    // Patient figure: body (part 0), head (part 1), floating status marker (part 2).
    MeshBuilder pm;
    pm.cylinder({0, 0, 0}, 0.26f, 1.1f, 10, {1, 1, 1}, 0.f);
    pm.sphere({0, 1.32f, 0}, 0.2f, 8, 6, {1, 1, 1}, 1.f);
    pm.box({0, 1.72f, 0}, {0.16f, 0.16f, 0.16f}, {1, 1, 1}, 2.f);
    patientMesh_.upload(pm, true, patientVbo_);
    MeshBuilder sm;
    sm.cylinder({0, 0, 0}, 0.3f, 1.25f, 10, {1, 1, 1}, 0.f);
    sm.sphere({0, 1.48f, 0}, 0.21f, 8, 6, {1, 1, 1}, 1.f);
    sm.box({0, 1.02f, 0}, {0.62f, 0.12f, 0.62f}, {1, 1, 1}, 2.f);  // "badge" band
    staffMesh_.upload(sm, true, staffVbo_);
    return true;
}

void Renderer::shutdown() {
    scene_.destroy();
    patientMesh_.destroy();
    staffMesh_.destroy();
    if (patientVbo_) glDeleteBuffers(1, &patientVbo_);
    if (staffVbo_) glDeleteBuffers(1, &staffVbo_);
    if (sceneProg_) glDeleteProgram(sceneProg_);
    if (patientProg_) glDeleteProgram(patientProg_);
    patientVbo_ = staffVbo_ = sceneProg_ = patientProg_ = 0;
}

bool Renderer::reserveInstances(int n) {
    if (n <= patientCapacity_) return false;
    patientCapacity_ = std::max(n, 1024);
    glBindBuffer(GL_ARRAY_BUFFER, patientVbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)patientCapacity_ * sizeof(RenderInstance), nullptr, GL_DYNAMIC_DRAW);
    return true;
}

void Renderer::uploadPatients(const std::vector<RenderInstance>& inst) {
    reserveInstances((int)inst.size());
    glBindBuffer(GL_ARRAY_BUFFER, patientVbo_);
    if (!inst.empty()) glBufferSubData(GL_ARRAY_BUFFER, 0, inst.size() * sizeof(RenderInstance), inst.data());
    patientCount_ = (int)inst.size();
}

void Renderer::uploadStaff(const std::vector<StaffInstance>& staff) {
    glBindBuffer(GL_ARRAY_BUFFER, staffVbo_);
    if ((int)staff.size() > staffCapacity_) {
        staffCapacity_ = (int)staff.size() * 2;
        glBufferData(GL_ARRAY_BUFFER, staffCapacity_ * sizeof(StaffInstance), nullptr, GL_DYNAMIC_DRAW);
    }
    if (!staff.empty()) glBufferSubData(GL_ARRAY_BUFFER, 0, staff.size() * sizeof(StaffInstance), staff.data());
    staffCount_ = (int)staff.size();
}

// ---------------------------------------------------------------- the scene ----
void Renderer::buildScene(const HospitalLayout& L) {
    MeshBuilder m;
    const glm::vec3 wallCol(0.93f, 0.93f, 0.95f), bedFrame(0.75f, 0.77f, 0.80f), sheet(0.97f, 0.98f, 1.0f);
    const float wallH = 1.4f, wallT = 0.25f, doorW = 4.0f;
    // ground, road, ambulance bay
    m.box({0, -0.06f, 0}, {320, 0.1f, 220}, {0.42f, 0.55f, 0.38f});
    m.box({-71, -0.01f, 0}, {12, 0.04f, 120}, {0.30f, 0.31f, 0.33f});
    for (int i = -5; i <= 5; ++i) m.box({-71, 0.02f, i * 10.f}, {0.3f, 0.02f, 4}, {0.95f, 0.9f, 0.4f});
    m.box({-70.5f, 1.1f, -14}, {2.6f, 2.2f, 5.5f}, {0.95f, 0.95f, 0.95f});  // ambulance
    m.box({-70.5f, 1.2f, -14}, {2.65f, 0.4f, 5.55f}, {0.85f, 0.15f, 0.15f});
    m.box({-70.5f, 2.3f, -15.5f}, {1.0f, 0.2f, 0.6f}, {0.2f, 0.4f, 1.0f});

    auto wallSeg = [&](float x0, float z0, float x1, float z1) {
        const float len = std::hypot(x1 - x0, z1 - z0);
        if (len < 0.05f) return;
        const bool alongX = std::fabs(z1 - z0) < 1e-3f;
        m.box({(x0 + x1) * 0.5f, wallH * 0.5f, (z0 + z1) * 0.5f},
              alongX ? glm::vec3(len, wallH, wallT) : glm::vec3(wallT, wallH, len), wallCol);
    };
    // wall along an edge with an optional door gap centred at `door`
    auto edge = [&](float x0, float z0, float x1, float z1, bool hasDoor, float door) {
        if (!hasDoor) return wallSeg(x0, z0, x1, z1);
        if (std::fabs(z1 - z0) < 1e-3f) {
            wallSeg(x0, z0, door - doorW / 2, z0);
            wallSeg(door + doorW / 2, z0, x1, z0);
        } else {
            wallSeg(x0, z0, x0, door - doorW / 2);
            wallSeg(x0, door + doorW / 2, x0, z1);
        }
    };

    for (int d = 0; d < D_COUNT; ++d) {
        const DepartmentInfo& di = L.dept(d);
        const Rect& r = di.area;
        const glm::vec3 c(di.color[0], di.color[1], di.color[2]);
        m.box({(r.x0 + r.x1) / 2, 0.0f, (r.z0 + r.z1) / 2}, {r.w(), 0.06f, r.h()}, c);
        if (d == D_CORRIDOR) continue;
        const Vec2 door = di.door;
        edge(r.x0, r.z0, r.x1, r.z0, std::fabs(door.z - r.z0) < 0.01f, door.x);  // north
        edge(r.x0, r.z1, r.x1, r.z1, std::fabs(door.z - r.z1) < 0.01f, door.x);  // south
        edge(r.x0, r.z0, r.x0, r.z1, d == D_ENTRANCE, 0.f);                       // west (entrance doors)
        edge(r.x1, r.z0, r.x1, r.z1, std::fabs(door.x - r.x1) < 0.01f, door.z);  // east
    }
    // corridor centre line
    m.box({6, 0.04f, 0}, {112, 0.02f, 0.25f}, {0.95f, 0.85f, 0.35f});

    auto bed = [&](Vec2 p, bool icu) {
        m.box({p.x, 0.35f, p.z}, {1.0f, 0.12f, 2.1f}, bedFrame);
        m.box({p.x, 0.48f, p.z}, {0.92f, 0.16f, 2.0f}, sheet);
        m.box({p.x, 0.62f, p.z + 0.75f}, {0.7f, 0.12f, 0.4f}, {1, 1, 1});
        for (int k = 0; k < 4; ++k)
            m.box({p.x + ((k & 1) ? 0.42f : -0.42f), 0.15f, p.z + ((k & 2) ? 0.95f : -0.95f)}, {0.07f, 0.3f, 0.07f},
                  bedFrame * 0.7f);
        if (icu) {
            m.box({p.x + 0.8f, 0.9f, p.z + 0.9f}, {0.12f, 1.8f, 0.12f}, {0.5f, 0.5f, 0.55f});
            m.box({p.x + 0.8f, 1.7f, p.z + 0.9f}, {0.45f, 0.35f, 0.1f}, {0.1f, 0.15f, 0.2f});
        }
    };
    for (const Vec2& p : L.erBeds) bed(p, false);
    for (const Vec2& p : L.icuBeds) bed(p, true);
    for (const Vec2& p : L.wardBeds) bed(p, false);
    for (size_t i = 0; i < L.triageDesks.size(); ++i) {
        const Vec2 d = {L.triageDesks[i].x, L.triageDesks[i].z - 1.0f};
        m.box({d.x, 0.45f, d.z}, {1.6f, 0.9f, 0.7f}, {0.55f, 0.40f, 0.28f});
        m.box({d.x, 0.3f, d.z + 1.0f}, {0.55f, 0.6f, 0.55f}, {0.3f, 0.45f, 0.7f});  // patient chair
        m.box({d.x + 0.4f, 1.05f, d.z}, {0.5f, 0.3f, 0.05f}, {0.1f, 0.1f, 0.12f});   // screen
    }
    for (const Vec2& p : L.labStations) {
        m.box({p.x + 1.1f, 0.6f, p.z}, {1.0f, 1.2f, 1.4f}, {0.45f, 0.42f, 0.55f});
        m.cylinder({p.x + 1.1f, 1.2f, p.z}, 0.45f, 0.25f, 12, {0.8f, 0.8f, 0.85f});
        m.box({p.x, 0.25f, p.z}, {0.6f, 0.5f, 0.6f}, {0.3f, 0.45f, 0.7f});
    }
    // waiting room chairs (the zone's first rows)
    const ZoneInfo& wz = L.zone(Z_WAITING_ROOM);
    for (int i = 0; i < 200; ++i) {
        const Vec2 s = L.zoneSlot(Z_WAITING_ROOM, i);
        if (s.z > wz.area.z1) break;
        m.box({s.x, 0.22f, s.z}, {0.55f, 0.44f, 0.5f}, {0.25f, 0.4f, 0.65f});
    }
    // ER doctor station counter
    m.box({-12, 0.55f, -5.5f}, {22, 1.1f, 0.6f}, {0.55f, 0.6f, 0.68f});
    // discharge exit gate
    m.box({59.5f, 1.4f, -10}, {0.4f, 2.8f, 6}, {0.35f, 0.75f, 0.55f});

    scene_.upload(m, false, 0);
}

void Renderer::render(const Camera& cam, int w, int h, float time, const RenderSettings& s) {
    glViewport(0, 0, w, h);
    const glm::vec3 fog(0.62f, 0.70f, 0.80f);
    glClearColor(fog.r, fog.g, fog.b, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    const glm::mat4 vp = cam.projection((float)w / std::max(1, h)) * cam.view();
    const glm::vec3 camPos = cam.position();
    const glm::vec3 light = glm::normalize(glm::vec3(0.4f, 1.0f, 0.3f));

    glUseProgram(sceneProg_);
    glUniformMatrix4fv(glGetUniformLocation(sceneProg_, "uViewProj"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform3fv(glGetUniformLocation(sceneProg_, "uCamPos"), 1, glm::value_ptr(camPos));
    glUniform3fv(glGetUniformLocation(sceneProg_, "uLightDir"), 1, glm::value_ptr(light));
    glUniform3fv(glGetUniformLocation(sceneProg_, "uFogColor"), 1, glm::value_ptr(fog));
    glUniform1f(glGetUniformLocation(sceneProg_, "uEmissive"), 0.f);
    glBindVertexArray(scene_.vao);
    glDrawElements(GL_TRIANGLES, scene_.count, GL_UNSIGNED_INT, nullptr);

    static const float prioCol[12] = {0.90f, 0.12f, 0.12f, 1.0f, 0.55f, 0.05f,
                                      0.98f, 0.88f, 0.10f, 0.20f, 0.75f, 0.25f};
    static const float stateCol[ST_COUNT * 3] = {
        0.5f, 0.5f, 0.5f,   0.75f, 0.75f, 0.80f,  // not arrived, arriving
        1.0f, 0.95f, 0.3f,  0.3f, 0.8f, 1.0f,     // wait triage, in triage
        1.0f, 0.75f, 0.3f,  0.65f, 0.4f, 1.0f,    // wait diag, in diag
        1.0f, 0.45f, 0.2f,  0.15f, 0.55f, 1.0f,   // wait ER, in treatment
        1.0f, 0.2f, 0.6f,   0.9f, 0.3f, 0.9f,     // boarding ICU, boarding ward
        0.95f, 0.1f, 0.2f,  0.2f, 0.85f, 0.4f,    // in ICU, in ward
        0.3f, 1.0f, 0.75f,  0.5f, 0.5f, 0.5f};    // discharging, discharged
    glUseProgram(patientProg_);
    glUniformMatrix4fv(glGetUniformLocation(patientProg_, "uViewProj"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform3fv(glGetUniformLocation(patientProg_, "uCamPos"), 1, glm::value_ptr(camPos));
    glUniform3fv(glGetUniformLocation(patientProg_, "uLightDir"), 1, glm::value_ptr(light));
    glUniform3fv(glGetUniformLocation(patientProg_, "uFogColor"), 1, glm::value_ptr(fog));
    glUniform3fv(glGetUniformLocation(patientProg_, "uPriorityColor"), 4, prioCol);
    glUniform3fv(glGetUniformLocation(patientProg_, "uStateColor"), ST_COUNT, stateCol);
    glUniform1f(glGetUniformLocation(patientProg_, "uTime"), time);
    glUniform1i(glGetUniformLocation(patientProg_, "uSelected"), s.selected);
    glUniform1i(glGetUniformLocation(patientProg_, "uHidePriorityMask"), s.hidePriorityMask);
    glUniform1i(glGetUniformLocation(patientProg_, "uMode"), 0);
    if (patientCount_ > 0) {
        glBindVertexArray(patientMesh_.vao);
        glDrawElementsInstanced(GL_TRIANGLES, patientMesh_.count, GL_UNSIGNED_INT, nullptr, patientCount_);
    }
    if (s.showStaff && staffCount_ > 0) {
        glUniform1i(glGetUniformLocation(patientProg_, "uMode"), 1);
        glUniform1i(glGetUniformLocation(patientProg_, "uSelected"), -1);
        glBindVertexArray(staffMesh_.vao);
        glDrawElementsInstanced(GL_TRIANGLES, staffMesh_.count, GL_UNSIGNED_INT, nullptr, staffCount_);
    }
    glBindVertexArray(0);
}

}  // namespace hs
