#include "Hospital.h"

#include <algorithm>
#include <cmath>

namespace hs {

HospitalLayout::HospitalLayout() {
    // name, area {x0,z0,x1,z1}, door, colour
    depts_[D_ENTRANCE] = {"Emergency Entrance", {-62, -12, -50, 12}, {-50, 0}, {0.55f, 0.62f, 0.72f}};
    depts_[D_TRIAGE] = {"Triage", {-48, -22, -32, -3}, {-40, -3}, {0.55f, 0.75f, 0.85f}};
    depts_[D_WAITING] = {"ER Waiting Room", {-48, 3, -30, 24}, {-39, 3}, {0.80f, 0.78f, 0.62f}};
    depts_[D_ER] = {"Emergency Room (ER)", {-28, -30, 4, -3}, {-12, -3}, {0.88f, 0.62f, 0.60f}};
    depts_[D_DIAG] = {"Diagnostics / Laboratory", {-26, 3, 0, 22}, {-13, 3}, {0.70f, 0.62f, 0.85f}};
    depts_[D_ICU] = {"Intensive Care Unit (ICU)", {8, -28, 32, -3}, {20, -3}, {0.92f, 0.50f, 0.55f}};
    depts_[D_WARD] = {"General Ward", {4, 3, 60, 32}, {32, 3}, {0.60f, 0.82f, 0.64f}};
    depts_[D_DISCHARGE] = {"Discharge Area", {36, -18, 60, -3}, {48, -3}, {0.65f, 0.85f, 0.80f}};
    depts_[D_CORRIDOR] = {"Corridor", {-50, -3, 62, 3}, {0, 0}, {0.78f, 0.78f, 0.80f}};

    // Triage queue lives in the entrance hall and overflows outside (west).
    zones_[Z_TRIAGE_QUEUE] = {D_ENTRANCE, {-57, -11, -51, 11}, 1.0f, {-1, 0}};
    // Waiting room overflows to the south (outside the building).
    zones_[Z_WAITING_ROOM] = {D_WAITING, {-47, 4, -31, 23}, 1.0f, {0, 1}};
    // Diagnostic queue: west half of the lab, overflows south.
    zones_[Z_DIAG_QUEUE] = {D_DIAG, {-25, 4, -15, 21}, 1.0f, {0, 1}};

    bounds_ = {-76, -32, 62, 34};

    // Route table: route(A,B) = doorA -> corridor(A) -> corridor(B) -> doorB.
    for (int a = 0; a < D_COUNT; ++a) {
        for (int b = 0; b < D_COUNT; ++b) {
            const int r = routeId(a, b);
            if (a == b || a == D_CORRIDOR || b == D_CORRIDOR) {
                routes_.len[r] = 0;
                continue;
            }
            const Vec2 da = depts_[a].door, db = depts_[b].door;
            const Vec2 pts[4] = {da, {da.x, 0.f}, {db.x, 0.f}, db};
            for (int k = 0; k < 4; ++k) {
                routes_.x[r][k] = pts[k].x;
                routes_.z[r][k] = pts[k].z;
            }
            routes_.len[r] = 4;
        }
    }
}

std::vector<Vec2> HospitalLayout::gridLayout(const Rect& r, int count, float margin) {
    std::vector<Vec2> out;
    if (count <= 0) return out;
    const float w = std::max(0.1f, r.w() - 2 * margin), h = std::max(0.1f, r.h() - 2 * margin);
    int cols = std::max(1, (int)std::ceil(std::sqrt(count * w / h)));
    int rows = (count + cols - 1) / cols;
    const float sx = w / cols, sz = h / rows;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        int c = i % cols, rr = i / cols;
        out.push_back({r.x0 + margin + sx * (c + 0.5f), r.z0 + margin + sz * (rr + 0.5f)});
    }
    return out;
}

void HospitalLayout::build(const Capacities& cap) {
    erBeds = gridLayout({-27, -29, 3, -8}, cap.erBeds, 0.8f);
    icuBeds = gridLayout({9, -27, 31, -5}, cap.icuBeds, 0.8f);
    wardBeds = gridLayout({5, 5, 59, 31}, cap.wardBeds, 0.8f);
    labStations = gridLayout({-12, 4, -1, 21}, cap.labSlots, 1.0f);
    doctorStations = gridLayout({-22, -7, -2, -4}, cap.doctors, 0.3f);
    const std::vector<Vec2> desks = gridLayout({-47, -21, -33, -12}, cap.nurses, 1.0f);
    triageDesks.clear();
    nursePositions.clear();
    for (const Vec2& d : desks) {
        triageDesks.push_back({d.x, d.z + 1.0f});      // patient chair (south of desk)
        nursePositions.push_back({d.x, d.z - 1.0f});  // nurse (north of desk)
    }
}

Vec2 HospitalLayout::zoneSlot(int zone, int index) const {
    const ZoneInfo& zi = zones_[zone];
    const float s = zi.spacing;
    if (zi.grow.z != 0) {  // rows run along x, overflow along z
        int cols = std::max(1, (int)(zi.area.w() / s));
        int c = index % cols, r = index / cols;
        float x = zi.area.x0 + s * (c + 0.5f);
        float z = (zi.grow.z > 0) ? zi.area.z0 + s * (r + 0.5f) : zi.area.z1 - s * (r + 0.5f);
        return {x, z};
    }
    int cols = std::max(1, (int)(zi.area.h() / s));  // rows run along z, overflow along x
    int c = index % cols, r = index / cols;
    float z = zi.area.z0 + s * (c + 0.5f);
    float x = (zi.grow.x > 0) ? zi.area.x0 + s * (r + 0.5f) : zi.area.x1 - s * (r + 0.5f);
    return {x, z};
}

Vec2 HospitalLayout::spawnPoint(int32_t id) const {
    // Ambulance bay / street outside the west entrance.
    return {-74.f + 6.f * hashUnit((uint64_t)id * 2 + 1), -8.f + 16.f * hashUnit((uint64_t)id * 2 + 2)};
}

Vec2 HospitalLayout::dischargePoint(int32_t id) const {
    return {50.f + 9.f * hashUnit((uint64_t)id * 7 + 3), -17.f + 12.f * hashUnit((uint64_t)id * 7 + 4)};
}

}  // namespace hs
