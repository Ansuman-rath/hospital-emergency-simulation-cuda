// Camera.h - top-down, orbit, free-look and follow-patient cameras.
#pragma once
#include <glm/glm.hpp>

namespace hs {

enum class CameraMode : int { TopDown = 0, Orbit, Free, Follow, Count };
const char* cameraModeName(CameraMode m);

struct Ray {
    glm::vec3 origin, dir;
};

class Camera {
public:
    CameraMode mode = CameraMode::Orbit;
    // orbit / top-down / follow parameters
    glm::vec3 target{-4.f, 0.f, 4.f};
    float yaw = -90.f;     // degrees, around +y
    float pitch = 50.f;    // degrees above the floor
    float distance = 165.f;
    // free-look parameters
    glm::vec3 freePos{-80.f, 25.f, 60.f};
    float freeYaw = -35.f, freePitch = -20.f;
    float fovDeg = 45.f;

    glm::mat4 view() const;
    glm::mat4 projection(float aspect) const;
    glm::vec3 position() const;
    glm::vec3 forward() const;

    // input
    void rotate(float dx, float dy);  // mouse delta in pixels
    void pan(float dx, float dy);     // mouse delta in pixels
    void zoom(float wheel);
    void moveFree(const glm::vec3& localDir, float dt, bool fast);  // x=right, y=up, z=forward
    void panKeys(const glm::vec2& dir, float dt, bool fast);
    void follow(const glm::vec3& p, float dt);
    void setMode(CameraMode m);

    Ray screenRay(float mx, float my, int w, int h) const;
};

}  // namespace hs
