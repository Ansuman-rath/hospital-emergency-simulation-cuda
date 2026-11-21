// Dashboard.h - Dear ImGui control panel, live metrics, performance and experiments.
#pragma once
#include <glm/glm.hpp>

namespace hs {
class App;
void drawDashboard(App& app);
void drawOverlay(App& app, const glm::mat4& viewProj);  // 3D labels, legend, status bar
}  // namespace hs
