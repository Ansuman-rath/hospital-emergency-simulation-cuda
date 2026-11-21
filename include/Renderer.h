// Renderer.h - OpenGL 3.3 core renderer: static hospital geometry + instanced patients.
// Rendering is fully separated from the simulation: it only consumes RenderInstance
// arrays (or a VBO written directly by CUDA via interop).
#pragma once
#include <glad/gl.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "Camera.h"
#include "Hospital.h"
#include "Patient.h"

namespace hs {

struct Vertex {
    glm::vec3 pos, normal, color;
    float part;
};

class MeshBuilder {
public:
    std::vector<Vertex> v;
    std::vector<uint32_t> idx;
    void box(const glm::vec3& center, const glm::vec3& size, const glm::vec3& color, float part = 0.f);
    void cylinder(const glm::vec3& base, float radius, float height, int seg, const glm::vec3& color, float part = 0.f);
    void sphere(const glm::vec3& c, float r, int seg, int rings, const glm::vec3& color, float part = 0.f);
};

struct GpuMesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei count = 0;
    void upload(const MeshBuilder& m, bool instanced, GLuint instanceVbo);
    void destroy();
};

struct StaffInstance {
    float x, z;
    uint32_t kind;  // 0 busy doctor, 1 idle doctor, 2 busy nurse, 3 idle nurse
};

struct RenderSettings {
    int selected = -1;
    int hidePriorityMask = 0;
    bool showStaff = true;
};

class Renderer {
public:
    bool init(const std::string& shaderDir, std::string* error);
    void shutdown();
    void buildScene(const HospitalLayout& layout);
    // Ensures the instance VBO can hold n patients; returns true if it was (re)allocated.
    bool reserveInstances(int n);
    void uploadPatients(const std::vector<RenderInstance>& inst);
    void setPatientCount(int n) { patientCount_ = n; }
    void uploadStaff(const std::vector<StaffInstance>& staff);
    GLuint patientInstanceBuffer() const { return patientVbo_; }
    void render(const Camera& cam, int w, int h, float time, const RenderSettings& s);

private:
    GLuint compile(const std::string& vsPath, const std::string& fsPath, std::string* err);
    GLuint sceneProg_ = 0, patientProg_ = 0;
    GpuMesh scene_, patientMesh_, staffMesh_;
    GLuint patientVbo_ = 0, staffVbo_ = 0;
    int patientCapacity_ = 0, patientCount_ = 0, staffCount_ = 0, staffCapacity_ = 0;
};

}  // namespace hs
