#include "SimulationEngine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "PatientGenerator.h"
#include "PatientUpdate.h"

namespace hs {

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

const char* queueName(int q) {
    static const char* n[Q_COUNT] = {"Triage", "Diagnostics", "ER (doctor + bed)", "ICU admission",
                                     "Ward admission"};
    return (q >= 0 && q < Q_COUNT) ? n[q] : "?";
}

const char* stateName(int s) {
    static const char* n[ST_COUNT] = {"Not arrived", "Arriving",     "Waiting triage", "In triage",
                                      "Waiting diag", "In diagnostics", "Waiting ER",  "In treatment",
                                      "Boarding->ICU", "Boarding->Ward", "In ICU",     "In ward",
                                      "Discharging",  "Discharged"};
    return (s >= 0 && s < ST_COUNT) ? n[s] : "?";
}

const char* priorityName(int p) {
    static const char* n[kNumPriorities] = {"P1 Critical", "P2 Serious", "P3 Moderate", "P4 Minor"};
    return (p >= 0 && p < kNumPriorities) ? n[p] : "?";
}

// ------------------------------------------------------------------ setup ----
void SimulationEngine::reset(const ScenarioConfig& cfg, std::unique_ptr<PatientBackend> backend) {
    cfg_ = cfg;
    cfg_.model.dt = cfg_.dt;
    layout_.build(cfg_.cap);
    res_.init(cfg_.cap);
    rec_ = PatientGenerator::generate(cfg_);
    const int n = (int)rec_.size();
    for (auto& q : queues_) q.init(n);
    for (auto& z : zones_) z.reset();
    dirty_.clear();
    cmds_.clear();
    events_.clear();
    time_ = 0;
    stepIndex_ = 0;
    nextArrival_ = 0;
    seq_ = 0;
    std::fill(std::begin(stateCounts_), std::end(stateCounts_), 0);
    stateCounts_[ST_NOT_ARRIVED] = n;
    std::fill(std::begin(queueWaitTotal_), std::end(queueWaitTotal_), 0.0);
    std::fill(std::begin(queueWaitMax_), std::end(queueWaitMax_), 0.0);
    std::fill(std::begin(queueServed_), std::end(queueServed_), 0);
    sumTreatDelay_ = sumLos_ = 0;
    treatStarted_ = escalations_ = 0;
    series_.clear();
    profile_ = {};
    transitions.clear();

    std::vector<PatientAoS> init(n);
    for (int i = 0; i < n; ++i) {
        const Vec2 sp = layout_.spawnPoint(i);
        init[i] = {sp.x, sp.z, sp.x, sp.z, rec_[i].speed, 0.f, 0.f, rec_[i].initHealth,
                   ST_NOT_ARRIVED, rec_[i].truePriority, -1, 0, 0};
    }
    backend_ = std::move(backend);
    backend_->init(init, cfg_.model, layout_.routes());

    activateArrivals();
    allocate();
    flushCommands();
    sampleSeries();
}

// ------------------------------------------------------------------- step ----
uint64_t SimulationEngine::sampleEvery() const {
    return std::max<uint64_t>(1, (uint64_t)std::lround(5.0 / cfg_.dt));
}

void SimulationEngine::step() {
    auto t0 = Clock::now();
    backend_->step(events_);  // parallel per-patient update (CPU or GPU)
    afterBackend(1, t0, Clock::now());
}

int SimulationEngine::stepBatch(int maxSteps) {
    const int m = quietSteps(std::max(1, maxSteps));
    auto t0 = Clock::now();
    const int done = m > 1 ? backend_->stepMulti(m, events_) : (backend_->step(events_), 1);
    afterBackend(done, t0, Clock::now());
    return done;
}

// Number of upcoming steps during which, if no patient event occurs, the controller has
// no work: bounded by the next scheduled arrival, the next sampling point and the limit.
int SimulationEngine::quietSteps(int limit) const {
    int64_t m = limit;
    const uint64_t s0 = stepIndex_;
    if (nextArrival_ < (int)rec_.size()) {
        const double a = rec_[nextArrival_].arrivalTime;
        uint64_t s = (uint64_t)std::max(0.0, std::floor(a / cfg_.dt));
        while (s > s0 + 1 && stepTime(s - 1) >= a) --s;  // first step index with time >= a
        while (stepTime(s) < a) ++s;
        if (s <= s0) s = s0 + 1;
        m = std::min<int64_t>(m, (int64_t)(s - s0));
    }
    const uint64_t every = sampleEvery();
    m = std::min<int64_t>(m, (int64_t)((s0 / every + 1) * every - s0));
    const double tmax = cfg_.maxSimHours * 60.0;
    uint64_t smax = (uint64_t)std::ceil(tmax / cfg_.dt);
    while (smax > 0 && stepTime(smax - 1) >= tmax) --smax;
    while (stepTime(smax) < tmax) ++smax;
    if (smax > s0) m = std::min<int64_t>(m, (int64_t)(smax - s0));
    return (int)std::max<int64_t>(1, m);
}

void SimulationEngine::afterBackend(int steps, Clock::time_point t0, Clock::time_point t1) {
    stepIndex_ += (uint64_t)steps;
    time_ = stepTime(stepIndex_);
    res_.accumulate((double)cfg_.dt * steps);  // utilisation over the interval just simulated

    for (const SimEvent& e : events_) processEvent(e);  // events are sorted by id
    activateArrivals();
    allocate();
    auto t2 = Clock::now();
    flushCommands();
    auto t3 = Clock::now();

    profile_.backendMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    profile_.controllerMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
    profile_.commandMs += std::chrono::duration<double, std::milli>(t3 - t2).count();
    profile_.steps += (uint64_t)steps;
    profile_.batches++;
    profile_.events += events_.size();
    if (stepIndex_ % sampleEvery() == 0) sampleSeries();
}

int SimulationEngine::advance(double simMinutes, double maxWallMs) {
    auto t0 = Clock::now();
    const int target = (int)std::floor(simMinutes / cfg_.dt + 1e-9);
    int done = 0;
    while (done < target && !finished()) {
        if (batching) done += stepBatch(target - done);
        else {
            step();
            ++done;
        }
        if (maxWallMs > 0 && msSince(t0) > maxWallMs) break;
    }
    return done;
}

void SimulationEngine::runToCompletion(double maxSimMinutes) {
    const double limit = maxSimMinutes > 0 ? maxSimMinutes : cfg_.maxSimHours * 60.0;
    while (!finished() && time_ < limit) {
        if (batching) {
            const double left = (limit - time_) / cfg_.dt;
            stepBatch((int)std::min(1e6, std::max(1.0, std::ceil(left - 1e-9))));
        } else {
            step();
        }
    }
}

bool SimulationEngine::finished() const {
    return stateCounts_[ST_DISCHARGED] == (int)rec_.size() || time_ >= cfg_.maxSimHours * 60.0;
}

// ---------------------------------------------------------------- helpers ----
void SimulationEngine::markDirty(PatientRecord& r, int mask) {
    r.cmdMask |= mask;
    if (!r.dirty) {
        r.dirty = true;
        dirty_.push_back(r.id);
    }
}

void SimulationEngine::setState(PatientRecord& r, int32_t s) {
    if (logTransitions) transitions.push_back({r.id, r.state, s, (float)time_});
    stateCounts_[r.state]--;
    stateCounts_[s]++;
    r.state = s;
    if (!inMask(kServiceMask, s)) r.svc = 0.f;
    markDirty(r);
}

void SimulationEngine::routeTo(PatientRecord& r, int dept, Vec2 target) {
    r.route = HospitalLayout::routeId(r.dept, dept);
    r.dept = dept;
    r.tx = target.x;
    r.tz = target.z;
    r.flags |= FLAG_MOVING;
    markDirty(r, CMD_ROUTE);
}

void SimulationEngine::enqueue(PatientRecord& r, int q, int zone) {
    if (zone >= 0) {
        r.zone = zone;
        r.zoneSlot = zones_[zone].acquire();
        routeTo(r, layout_.zone(zone).dept, layout_.zoneSlot(zone, r.zoneSlot));
    }
    queues_[q].push(r.id, q == Q_TRIAGE ? 0 : r.priority, seq_++);
    r.enqueueTime = time_;
}

void SimulationEngine::dequeue(PatientRecord& r, int q) {
    const double w = time_ - r.enqueueTime;
    r.cpuWaitTotal += w;
    queueWaitTotal_[q] += w;
    queueWaitMax_[q] = std::max(queueWaitMax_[q], w);
    queueServed_[q]++;
    if (r.zone >= 0) {
        zones_[r.zone].release(r.zoneSlot);
        r.zone = r.zoneSlot = -1;
    }
}

void SimulationEngine::releaseRes(PatientRecord& r, int pool, int32_t& slot) {
    res_.pool(pool).release(slot, r.id);
    slot = -1;
}

void SimulationEngine::discharge(PatientRecord& r) {
    // Invariant 3: a patient cannot be discharged before completing treatment.
    if (!r.treatmentDone) throw std::logic_error("discharge before treatment completed");
    setState(r, ST_DISCHARGING);
    routeTo(r, D_DISCHARGE, layout_.dischargePoint(r.id));
}

// ----------------------------------------------------------------- events ----
void SimulationEngine::processEvent(const SimEvent& e) {
    PatientRecord& r = rec_[e.id];
    if (e.bits & EV_ESCALATED) {
        r.priority = e.priority;
        r.escalations++;
        escalations_++;
        queues_[Q_DIAG].updatePriority(r.id, r.priority);
        queues_[Q_ER].updatePriority(r.id, r.priority);
    }
    if (e.bits & EV_ARRIVED) {
        r.flags &= ~FLAG_MOVING;
        if (r.state == ST_ARRIVING) {
            setState(r, ST_WAIT_TRIAGE);
            enqueue(r, Q_TRIAGE, -1);  // keeps the queue slot it walked to
        } else if (r.state == ST_DISCHARGING) {
            setState(r, ST_DISCHARGED);
            r.flags = 0;
            r.dischargeTime = time_;
            sumLos_ += time_ - r.arrivalTime;
        }
    }
    if (e.bits & EV_SERVICE_DONE) {
        r.svc = 0.f;
        switch (r.state) {
            case ST_IN_TRIAGE:
                releaseRes(r, R_NURSE, r.triageDesk);
                r.flags |= FLAG_TRIAGED;
                r.priority = r.truePriority;
                if (r.needsDiag) {
                    setState(r, ST_WAIT_DIAG);
                    enqueue(r, Q_DIAG, Z_DIAG_QUEUE);
                } else {
                    setState(r, ST_WAIT_ER);
                    enqueue(r, Q_ER, Z_WAITING_ROOM);
                }
                break;
            case ST_IN_DIAG:
                releaseRes(r, R_LAB, r.labSlot);
                setState(r, ST_WAIT_ER);
                enqueue(r, Q_ER, Z_WAITING_ROOM);
                break;
            case ST_IN_TREATMENT:
                releaseRes(r, R_DOCTOR, r.doctor);
                r.treatmentDone = true;
                r.treatmentEnd = time_;
                if (r.disposition == DISP_ICU) {
                    setState(r, ST_WAIT_ICU);  // boards in the ER bed
                    enqueue(r, Q_ICU, -1);
                } else if (r.disposition == DISP_WARD) {
                    setState(r, ST_WAIT_WARD);
                    enqueue(r, Q_WARD, -1);
                } else {
                    releaseRes(r, R_ER_BED, r.erBed);
                    discharge(r);
                }
                break;
            case ST_IN_ICU:
                if (r.icuStepDown) {
                    setState(r, ST_WAIT_WARD);  // boards in the ICU bed
                    enqueue(r, Q_WARD, -1);
                } else {
                    releaseRes(r, R_ICU_BED, r.icuBed);
                    discharge(r);
                }
                break;
            case ST_IN_WARD:
                releaseRes(r, R_WARD_BED, r.wardBed);
                discharge(r);
                break;
            default: throw std::logic_error("service-done event in non-service state");
        }
    }
}

void SimulationEngine::activateArrivals() {
    const int n = (int)rec_.size();
    while (nextArrival_ < n && rec_[nextArrival_].arrivalTime <= time_) {
        PatientRecord& r = rec_[nextArrival_++];
        r.dept = D_ENTRANCE;
        r.priority = r.truePriority;
        markDirty(r, CMD_INIT);
        if (r.ambulance && r.truePriority == P1) {
            // Pre-triaged critical ambulance patient: fast track straight to the ER queue.
            r.flags = FLAG_TRIAGED | FLAG_AMBULANCE;
            setState(r, ST_WAIT_ER);
            enqueue(r, Q_ER, Z_WAITING_ROOM);
        } else {
            r.flags = r.ambulance ? FLAG_AMBULANCE : 0;
            setState(r, ST_ARRIVING);
            r.zone = Z_TRIAGE_QUEUE;
            r.zoneSlot = zones_[Z_TRIAGE_QUEUE].acquire();
            routeTo(r, D_ENTRANCE, layout_.zoneSlot(Z_TRIAGE_QUEUE, r.zoneSlot));
        }
    }
}

// ------------------------------------------------------------- allocation ----
// Serial, deterministic, downstream-first so beds freed this step can be reused
// upstream in the same step. Queue order = (priority, FIFO sequence, id).
void SimulationEngine::allocate() {
    ResourcePool& ward = res_.pool(R_WARD_BED);
    while (!queues_[Q_WARD].empty() && ward.available() > 0) {
        PatientRecord& r = rec_[queues_[Q_WARD].popFront()];
        dequeue(r, Q_WARD);
        if (r.erBed >= 0) releaseRes(r, R_ER_BED, r.erBed);
        if (r.icuBed >= 0) releaseRes(r, R_ICU_BED, r.icuBed);
        r.wardBed = ward.acquire(r.id);
        setState(r, ST_IN_WARD);
        r.svc = r.wardDur;
        routeTo(r, D_WARD, layout_.wardBeds[r.wardBed]);
    }
    ResourcePool& icu = res_.pool(R_ICU_BED);
    while (!queues_[Q_ICU].empty() && icu.available() > 0) {
        PatientRecord& r = rec_[queues_[Q_ICU].popFront()];
        dequeue(r, Q_ICU);
        releaseRes(r, R_ER_BED, r.erBed);
        r.icuBed = icu.acquire(r.id);
        setState(r, ST_IN_ICU);
        r.svc = r.icuDur;
        routeTo(r, D_ICU, layout_.icuBeds[r.icuBed]);
    }
    ResourcePool& nurse = res_.pool(R_NURSE);
    while (!queues_[Q_TRIAGE].empty() && nurse.available() > 0) {
        PatientRecord& r = rec_[queues_[Q_TRIAGE].popFront()];
        dequeue(r, Q_TRIAGE);
        r.triageDesk = nurse.acquire(r.id);
        setState(r, ST_IN_TRIAGE);
        r.svc = r.triageDur;
        routeTo(r, D_TRIAGE, layout_.triageDesks[r.triageDesk]);
    }
    ResourcePool& lab = res_.pool(R_LAB);
    while (!queues_[Q_DIAG].empty() && lab.available() > 0) {
        PatientRecord& r = rec_[queues_[Q_DIAG].popFront()];
        dequeue(r, Q_DIAG);
        r.labSlot = lab.acquire(r.id);
        setState(r, ST_IN_DIAG);
        r.svc = r.diagDur;
        routeTo(r, D_DIAG, layout_.labStations[r.labSlot]);
    }
    ResourcePool& doc = res_.pool(R_DOCTOR);
    ResourcePool& er = res_.pool(R_ER_BED);
    while (!queues_[Q_ER].empty() && doc.available() > 0 && er.available() > 0) {
        PatientRecord& r = rec_[queues_[Q_ER].popFront()];
        dequeue(r, Q_ER);
        r.doctor = doc.acquire(r.id);
        r.erBed = er.acquire(r.id);
        setState(r, ST_IN_TREATMENT);
        r.svc = r.treatDur;
        r.treatmentStart = time_;
        sumTreatDelay_ += time_ - r.arrivalTime;
        treatStarted_++;
        routeTo(r, D_ER, layout_.erBeds[r.erBed]);
    }
}

void SimulationEngine::flushCommands() {
    cmds_.clear();
    for (int32_t id : dirty_) {
        PatientRecord& r = rec_[id];
        const Vec2 sp = layout_.spawnPoint(id);
        cmds_.push_back({id, r.cmdMask, r.state, r.priority, r.flags, r.route, r.tx, r.tz, r.svc, sp.x, sp.z,
                         r.initHealth, r.speed});
        r.dirty = false;
        r.cmdMask = 0;
    }
    dirty_.clear();
    profile_.commands += cmds_.size();
    if (!cmds_.empty()) backend_->applyCommands(cmds_);
}

// ---------------------------------------------------------------- metrics ----
void SimulationEngine::sampleSeries() {
    if (series_.t.size() >= 20000) return;
    series_.t.push_back((float)time_);
    series_.queueTriage.push_back((float)queues_[Q_TRIAGE].size());
    series_.queueER.push_back((float)queues_[Q_ER].size());
    series_.queueBoarding.push_back((float)(queues_[Q_ICU].size() + queues_[Q_WARD].size()));
    auto occ = [&](int p) {
        const ResourcePool& rp = res_.pool(p);
        return rp.capacity() > 0 ? 100.f * rp.busy() / rp.capacity() : 0.f;
    };
    series_.erOcc.push_back(occ(R_ER_BED));
    series_.icuOcc.push_back(occ(R_ICU_BED));
    series_.wardOcc.push_back(occ(R_WARD_BED));
    series_.discharged.push_back((float)stateCounts_[ST_DISCHARGED]);
    series_.active.push_back((float)(rec_.size() - stateCounts_[ST_NOT_ARRIVED] - stateCounts_[ST_DISCHARGED]));
}

LiveMetrics SimulationEngine::computeMetrics() {
    LiveMetrics m;
    m.simTime = time_;
    m.total = (int)rec_.size();
    for (int s = 0; s < ST_COUNT; ++s) {
        m.stateCounts[s] = stateCounts_[s];
        if (inMask(kWaitMask, s)) m.waiting += stateCounts_[s];
        if (inMask(kServiceMask, s)) m.inService += stateCounts_[s];
    }
    m.discharged = stateCounts_[ST_DISCHARGED];
    m.arrived = m.total - stateCounts_[ST_NOT_ARRIVED];
    m.active = m.arrived - m.discharged;
    m.inTreatment = stateCounts_[ST_IN_TREATMENT];
    m.escalations = escalations_;
    for (int i = 0; i < nextArrival_; ++i) {
        const PatientRecord& r = rec_[i];
        if (!inMask(kActiveMask, r.state)) continue;
        if (r.flags & FLAG_TRIAGED) m.activeByPriority[r.priority]++;
        else m.untriaged++;
    }
    double curWaitSum = 0;
    double bestLen = 0, bestWait = 0;
    for (int q = 0; q < Q_COUNT; ++q) {
        QueueStats& qs = m.queues[q];
        qs.length = (int)queues_[q].size();
        qs.served = queueServed_[q];
        double sum = 0, mx = queueWaitMax_[q];
        for (const QueueKey& k : queues_[q].items()) {
            const double w = time_ - rec_[k.id].enqueueTime;
            sum += w;
            mx = std::max(mx, w);
        }
        curWaitSum += sum;
        const int n = qs.served + qs.length;
        qs.avgWait = n > 0 ? (queueWaitTotal_[q] + sum) / n : 0.0;
        qs.maxWait = mx;
        if (qs.length > bestLen) { bestLen = qs.length; m.largestQueue = q; }
        if (qs.avgWait > bestWait) { bestWait = qs.avgWait; m.slowestQueue = q; }
    }
    for (int p = 0; p < R_COUNT; ++p) {
        m.busy[p] = res_.pool(p).busy();
        m.capacity[p] = res_.pool(p).capacity();
        m.utilization[p] = res_.pool(p).utilization(time_);
    }
    const HotReduction h = backend_->reduce();
    m.avgWait = m.arrived > 0 ? h.sumWait / m.arrived : 0.0;
    m.maxWait = h.maxWait;
    m.avgHealth = h.activeCount > 0 ? h.sumHealth / h.activeCount : 0.0;
    double cpuWait = curWaitSum;
    for (int q = 0; q < Q_COUNT; ++q) cpuWait += queueWaitTotal_[q];
    m.avgWaitCpu = m.arrived > 0 ? cpuWait / m.arrived : 0.0;
    m.treatmentStarted = treatStarted_;
    m.avgTreatmentDelay = treatStarted_ > 0 ? sumTreatDelay_ / treatStarted_ : 0.0;
    m.avgLengthOfStay = m.discharged > 0 ? sumLos_ / m.discharged : 0.0;
    m.throughputPerHour = time_ > 0 ? m.discharged / (time_ / 60.0) : 0.0;
    return m;
}

bool SimulationEngine::checkInvariants(std::string* err) const {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        return false;
    };
    if (!res_.validate(err)) return false;
    auto holds = [&](int pool, int32_t slot, int32_t id) {
        return slot >= 0 && slot < res_.pool(pool).capacity() && res_.pool(pool).occupant(slot) == id;
    };
    for (const PatientRecord& r : rec_) {
        const std::string tag = "patient " + std::to_string(r.id) + " (" + stateName(r.state) + "): ";
        switch (r.state) {
            case ST_IN_TREATMENT:
                if (!holds(R_DOCTOR, r.doctor, r.id) || !holds(R_ER_BED, r.erBed, r.id))
                    return fail(tag + "treated without doctor+bed");
                break;
            case ST_WAIT_ICU:
                if (!holds(R_ER_BED, r.erBed, r.id)) return fail(tag + "boarding without ER bed");
                break;
            case ST_WAIT_WARD:
                if (!holds(R_ER_BED, r.erBed, r.id) && !holds(R_ICU_BED, r.icuBed, r.id))
                    return fail(tag + "boarding without bed");
                break;
            case ST_IN_ICU:
                if (!holds(R_ICU_BED, r.icuBed, r.id)) return fail(tag + "in ICU without bed");
                break;
            case ST_IN_WARD:
                if (!holds(R_WARD_BED, r.wardBed, r.id)) return fail(tag + "in ward without bed");
                break;
            case ST_IN_TRIAGE:
                if (!holds(R_NURSE, r.triageDesk, r.id)) return fail(tag + "triage without nurse");
                break;
            case ST_IN_DIAG:
                if (!holds(R_LAB, r.labSlot, r.id)) return fail(tag + "diagnostics without slot");
                break;
            case ST_DISCHARGING:
            case ST_DISCHARGED:
                if (!r.treatmentDone) return fail(tag + "discharged without treatment");
                if (r.doctor >= 0 || r.erBed >= 0 || r.icuBed >= 0 || r.wardBed >= 0 || r.triageDesk >= 0 ||
                    r.labSlot >= 0)
                    return fail(tag + "discharged while holding a resource");
                break;
            default: break;
        }
        if (r.doctor >= 0 && r.state != ST_IN_TREATMENT) return fail(tag + "holds a doctor outside treatment");
        const int expectQ = r.state == ST_WAIT_TRIAGE ? Q_TRIAGE
                            : r.state == ST_WAIT_DIAG ? Q_DIAG
                            : r.state == ST_WAIT_ER   ? Q_ER
                            : r.state == ST_WAIT_ICU  ? Q_ICU
                            : r.state == ST_WAIT_WARD ? Q_WARD
                                                      : -1;
        for (int q = 0; q < Q_COUNT; ++q)
            if (queues_[q].contains(r.id) != (q == expectQ)) return fail(tag + "queue membership mismatch");
    }
    return true;
}

}  // namespace hs
