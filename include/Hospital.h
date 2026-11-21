// Hospital.h - simplified virtual hospital floor plan: departments, doors, corridor,
// route table and positions of beds / desks / queue slots.
//
// Coordinate system: metres on the floor plane (x to the east, z to the south),
// y is up in the renderer. One main east-west corridor at z = 0 connects all
// departments; every route is door -> corridor -> corridor -> door.
#pragma once
#include <queue>
#include <string>
#include <vector>

#include "Patient.h"

namespace hs {

struct Vec2 {
    float x = 0, z = 0;
};

struct Rect {
    float x0 = 0, z0 = 0, x1 = 0, z1 = 0;
    float w() const { return x1 - x0; }
    float h() const { return z1 - z0; }
    Vec2 center() const { return {(x0 + x1) * 0.5f, (z0 + z1) * 0.5f}; }
};

enum Department : int32_t {
    D_ENTRANCE = 0,
    D_TRIAGE,
    D_WAITING,
    D_ER,
    D_DIAG,
    D_ICU,
    D_WARD,
    D_DISCHARGE,
    D_CORRIDOR,
    D_COUNT
};
static_assert(D_COUNT * D_COUNT <= kMaxRoutes, "route table too small");

// Waiting zones whose positions are handed out by a slot allocator.
enum QueueZone : int32_t { Z_TRIAGE_QUEUE = 0, Z_WAITING_ROOM, Z_DIAG_QUEUE, Z_COUNT };

struct DepartmentInfo {
    const char* name;
    Rect area;
    Vec2 door;
    float color[3];
};

struct ZoneInfo {
    int32_t dept;
    Rect area;
    float spacing;
    Vec2 grow;  // direction in which rows overflow when the area is full
};

struct Capacities {
    int doctors = 8;
    int nurses = 4;  // each on-duty triage nurse staffs one triage desk
    int erBeds = 16;
    int icuBeds = 8;
    int wardBeds = 40;
    int labSlots = 4;
};

// Lowest-free-index allocator: deterministic and O(log n).
class SlotAllocator {
public:
    void reset() {
        freed_ = {};
        next_ = 0;
        used_ = 0;
    }
    int acquire() {
        ++used_;
        if (!freed_.empty()) {
            int s = freed_.top();
            freed_.pop();
            return s;
        }
        return next_++;
    }
    void release(int s) {
        --used_;
        freed_.push(s);
    }
    int used() const { return used_; }

private:
    std::priority_queue<int, std::vector<int>, std::greater<int>> freed_;
    int next_ = 0;
    int used_ = 0;
};

class HospitalLayout {
public:
    HospitalLayout();
    void build(const Capacities& cap);

    const DepartmentInfo& dept(int d) const { return depts_[d]; }
    const ZoneInfo& zone(int z) const { return zones_[z]; }
    static int routeId(int from, int to) { return from * D_COUNT + to; }
    const RouteTable& routes() const { return routes_; }

    Vec2 zoneSlot(int zone, int index) const;
    Vec2 spawnPoint(int32_t patientId) const;
    Vec2 dischargePoint(int32_t patientId) const;
    Rect bounds() const { return bounds_; }

    // Positions where a patient lies / sits for each resource slot.
    std::vector<Vec2> erBeds, icuBeds, wardBeds, labStations, triageDesks, doctorStations;
    // Staff positions next to triage desks.
    std::vector<Vec2> nursePositions;

    static std::vector<Vec2> gridLayout(const Rect& r, int count, float margin);

private:
    DepartmentInfo depts_[D_COUNT];
    ZoneInfo zones_[Z_COUNT];
    RouteTable routes_{};
    Rect bounds_;
};

// Deterministic hash used for jitter / per-patient random streams.
inline uint64_t hashU64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
inline float hashUnit(uint64_t x) { return (float)(hashU64(x) >> 40) * (1.0f / 16777216.0f); }

}  // namespace hs
