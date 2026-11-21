// PatientUpdate.h - the per-patient update rules, compiled for BOTH the CPU and the GPU.
//
// Every function here is __host__ __device__ (HS_HD). The CPU baseline loops call
// exactly the same code that the CUDA kernels call, which guarantees that both
// implementations perform equivalent computations. CUDA is compiled with
// --fmad=false (no fused multiply-add contraction) and IEEE sqrt/div, and MSVC
// uses /fp:precise, so the CPU and GPU results are expected to be bit-identical;
// tests still compare with an explicit tolerance (see tests/test_main.cpp).
//
// Each function is independent per patient: no patient reads another patient's
// data, which is what makes "one CUDA thread per patient" valid for this phase.
#pragma once
#include <math.h>

#include "Patient.h"

namespace hs {

HS_HD bool inMask(uint32_t mask, int32_t state) { return ((mask >> state) & 1u) != 0u; }

// 1) Movement along the route waypoints and finally to the personal target.
HS_HD uint32_t moveStep(float& x, float& z, float tx, float tz, float speed, int32_t route, int32_t& wp,
                        int32_t& flags, float dt, const RouteTable& rt) {
    if ((flags & FLAG_MOVING) == 0) return 0u;
    float budget = speed * dt;
    const int32_t len = (route >= 0) ? rt.len[route] : 0;
    // A step may pass several (short) waypoints; bounded loop for the GPU.
    for (int iter = 0; iter <= kRouteLen; ++iter) {
        float gx, gz;
        if (wp < len) {
            gx = rt.x[route][wp];
            gz = rt.z[route][wp];
        } else {
            gx = tx;
            gz = tz;
        }
        const float dx = gx - x;
        const float dz = gz - z;
        const float d = sqrtf(dx * dx + dz * dz);
        if (d <= budget) {
            x = gx;
            z = gz;
            budget = budget - d;
            if (wp >= len) {
                flags &= ~FLAG_MOVING;
                return EV_ARRIVED;
            }
            ++wp;
        } else {
            const float s = budget / d;  // d > budget >= 0, so d > 0
            x = x + dx * s;
            z = z + dz * s;
            return 0u;
        }
    }
    return 0u;
}

// 2) Waiting-time accumulation.
HS_HD uint32_t waitStep(int32_t state, float& waitingTime, float dt) {
    if (inMask(kWaitMask, state)) waitingTime = waitingTime + dt;
    return 0u;
}

// 3) Treatment / service progress (starts once the patient reached the location).
HS_HD uint32_t serviceStep(int32_t state, int32_t flags, float& remaining, float dt) {
    if (inMask(kServiceMask, state) && (flags & FLAG_MOVING) == 0 && remaining > 0.0f) {
        remaining = remaining - dt;
        if (remaining <= 0.0f) {
            remaining = 0.0f;
            return EV_SERVICE_DONE;
        }
    }
    return 0u;
}

// 4) Independent patient-state calculation: synthetic deterioration / recovery and
//    priority escalation (e.g. a P3 patient waiting too long becomes P2).
HS_HD uint32_t healthStep(int32_t state, int32_t flags, int32_t& priority, float& health, const UpdateParams& p) {
    if (inMask(kDeteriorateMask, state)) {
        health = health - p.decay[priority] * p.dt;
        if (health < p.minHealth) health = p.minHealth;
        if ((flags & FLAG_TRIAGED) != 0 && priority > 0 && health < p.escalateBelow[priority]) {
            priority = priority - 1;
            return EV_ESCALATED;
        }
    } else if (inMask(kRecoverMask, state) && (flags & FLAG_MOVING) == 0) {
        health = health + p.recoverRate * p.dt;
        if (health > 100.0f) health = 100.0f;
    }
    return 0u;
}

// Fused update: all four operations in one pass over the patient's registers.
HS_HD uint32_t updatePatient(float& x, float& z, float tx, float tz, float speed, float& waitingTime,
                             float& remaining, float& health, int32_t state, int32_t& priority, int32_t route,
                             int32_t& wp, int32_t& flags, const UpdateParams& p, const RouteTable& rt) {
    if (!inMask(kActiveMask, state)) return 0u;
    uint32_t ev = moveStep(x, z, tx, tz, speed, route, wp, flags, p.dt, rt);
    ev |= waitStep(state, waitingTime, p.dt);
    ev |= serviceStep(state, flags, remaining, p.dt);
    ev |= healthStep(state, flags, priority, health, p);
    return ev;
}

HS_HD uint32_t updatePatientAoS(PatientAoS& q, const UpdateParams& p, const RouteTable& rt) {
    return updatePatient(q.x, q.z, q.tx, q.tz, q.speed, q.waitingTime, q.serviceRemaining, q.health, q.state,
                         q.priority, q.route, q.waypoint, q.flags, p, rt);
}

// Apply one CPU command to a patient (used by CPU backend and by the scatter kernel).
HS_HD void applyCommand(PatientAoS& q, const PatientCommand& c) {
    q.state = c.state;
    q.priority = c.priority;
    q.flags = c.flags;
    q.serviceRemaining = c.serviceRemaining;
    if (c.mask & CMD_ROUTE) {
        q.route = c.route;
        q.waypoint = 0;
        q.tx = c.tx;
        q.tz = c.tz;
    }
    if (c.mask & CMD_INIT) {
        q.x = c.x;
        q.z = c.z;
        q.health = c.health;
        q.speed = c.speed;
        q.waitingTime = 0.0f;
    }
}

}  // namespace hs
