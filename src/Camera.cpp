#include "Camera.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace hs {

const char* cameraModeName(CameraMode m) {
    switch (m) {
        case CameraMode::TopDown: return "Top-down";
        case CameraMode::Orbit: return "Orbit";
        case CameraMode::Free: return "Free-look";
        case CameraMode::Follow: return "Follow patient";
        default: return "?";
    }
}

static glm::vec3 dirFromAngles(float yawDeg, float pitchDeg) {
    const float y = glm::radians(yawDeg), p = glm::radians(pitchDeg);
    return {std::cos(p) * std::cos(y), std::sin(p), std::cos(p) * std::sin(y)};
}

glm::vec3 Camera::position() const {
    switch (mode) {
        case CameraMode::TopDown: return target + glm::vec3(0.f, distance, 0.001f);
        case CameraMode::Free: return freePos;
        default: return target - dirFromAngles(yaw, -pitch) * distance;
    }
}

glm::vec3 Camera::forward() const {
    if (mode == CameraMode::Free) return dirFromAngles(freeYaw, freePitch);
    return glm::normalize(target - position());
}

glm::mat4 Camera::view() const {
    if (mode == CameraMode::TopDown)
        return glm::lookAt(position(), target, glm::vec3(0.f, 0.f, -1.f));
    if (mode == CameraMode::Free)
        return glm::lookAt(freePos, freePos + dirFromAngles(freeYaw, freePitch), glm::vec3(0.f, 1.f, 0.f));
    return glm::lookAt(position(), target, glm::vec3(0.f, 1.f, 0.f));
}

glm::mat4 Camera::projection(float aspect) const {
    return glm::perspective(glm::radians(fovDeg), std::max(0.1f, aspect), 0.3f, 1500.f);
}

void Camera::rotate(float dx, float dy) {
    if (mode == CameraMode::Free) {
        freeYaw += dx * 0.15f;
        freePitch = std::clamp(freePitch - dy * 0.15f, -89.f, 89.f);
    } else if (mode != CameraMode::TopDown) {
        yaw += dx * 0.25f;
        pitch = std::clamp(pitch + dy * 0.25f, 5.f, 89.f);
    }
}

void Camera::pan(float dx, float dy) {
    if (mode == CameraMode::Free) return;
    const float s = distance * 0.0015f;
    glm::vec3 fwd, right;
    if (mode == CameraMode::TopDown) {
        fwd = {0, 0, -1};
        right = {1, 0, 0};
    } else {
        const glm::vec3 f = dirFromAngles(yaw, 0.f);
        fwd = {f.x, 0, f.z};
        right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    }
    target += (-right * dx + fwd * dy) * s;
}

void Camera::zoom(float wheel) {
    if (mode == CameraMode::Free) {
        freePos += dirFromAngles(freeYaw, freePitch) * wheel * 4.f;
        return;
    }
    distance = std::clamp(distance * std::pow(0.88f, wheel), 4.f, 600.f);
}

void Camera::moveFree(const glm::vec3& d, float dt, bool fast) {
    const glm::vec3 f = dirFromAngles(freeYaw, freePitch);
    const glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
    const float speed = (fast ? 90.f : 30.f) * dt;
    freePos += (r * d.x + glm::vec3(0, 1, 0) * d.y + f * d.z) * speed;
}

void Camera::panKeys(const glm::vec2& dir, float dt, bool fast) {
    // dir.x = right, dir.y = forward (screen-relative)
    const float px = (fast ? 900.f : 350.f) * dt;
    pan(-dir.x * px, dir.y * px);
}

void Camera::follow(const glm::vec3& p, float dt) {
    const float a = 1.f - std::exp(-dt * 6.f);
    target = glm::mix(target, p, a);
}

void Camera::setMode(CameraMode m) {
    if (m == mode) return;
    if (m == CameraMode::Free) {
        freePos = position();
        const glm::vec3 f = glm::normalize(target - freePos);
        freeYaw = glm::degrees(std::atan2(f.z, f.x));
        freePitch = glm::degrees(std::asin(std::clamp(f.y, -1.f, 1.f)));
    }
    if (m == CameraMode::Follow) distance = std::min(distance, 30.f);
    mode = m;
}

Ray Camera::screenRay(float mx, float my, int w, int h) const {
    const glm::mat4 inv = glm::inverse(projection((float)w / std::max(1, h)) * view());
    const float x = 2.f * mx / std::max(1, w) - 1.f, y = 1.f - 2.f * my / std::max(1, h);
    glm::vec4 a = inv * glm::vec4(x, y, -1.f, 1.f), b = inv * glm::vec4(x, y, 1.f, 1.f);
    a /= a.w;
    b /= b.w;
    return {glm::vec3(a), glm::normalize(glm::vec3(b - a))};
}

}  // namespace hs
