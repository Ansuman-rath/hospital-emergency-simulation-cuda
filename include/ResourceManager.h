// ResourceManager.h - exclusive hospital resources and priority queues (CPU side).
//
// Resource allocation has shared dependencies (two patients may want the same
// bed), so it is performed serially on the CPU in a fixed, deterministic order.
// This makes duplicate assignment impossible by construction: acquire() only
// hands out a slot whose occupant is -1, and release() verifies the occupant.
#pragma once
#include <cstdint>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Hospital.h"

namespace hs {

enum ResourceType : int { R_DOCTOR = 0, R_NURSE, R_ER_BED, R_ICU_BED, R_WARD_BED, R_LAB, R_COUNT };
const char* resourceName(int r);

class ResourcePool {
public:
    void init(const std::string& name, int capacity);
    int acquire(int32_t patientId);              // lowest free slot, or -1 if none
    void release(int slot, int32_t patientId);   // throws std::logic_error on mismatch
    int capacity() const { return (int)occ_.size(); }
    int busy() const { return busy_; }
    int available() const { return capacity() - busy_; }
    int32_t occupant(int slot) const { return occ_[slot]; }
    void accumulate(double dt) { busyTime_ += busy_ * dt; }
    double utilization(double elapsed) const {
        return (elapsed > 0 && capacity() > 0) ? busyTime_ / (elapsed * capacity()) : 0.0;
    }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::vector<int32_t> occ_;
    std::priority_queue<int, std::vector<int>, std::greater<int>> free_;
    int busy_ = 0;
    double busyTime_ = 0;
};

class ResourceManager {
public:
    void init(const Capacities& cap);
    ResourcePool& pool(int r) { return pools_[r]; }
    const ResourcePool& pool(int r) const { return pools_[r]; }
    void accumulate(double dt) {
        for (auto& p : pools_) p.accumulate(dt);
    }
    // Checks that no patient holds two slots of the same pool and busy counts match.
    bool validate(std::string* err) const;

private:
    ResourcePool pools_[R_COUNT];
};

// Priority queue with stable FIFO order inside a priority level and support for
// re-keying (priority escalation). Ordering key: (priority, enqueue sequence, id).
struct QueueKey {
    int32_t prio;
    int64_t seq;
    int32_t id;
    bool operator<(const QueueKey& o) const {
        if (prio != o.prio) return prio < o.prio;
        if (seq != o.seq) return seq < o.seq;
        return id < o.id;
    }
};

class PatientQueue {
public:
    void init(int maxPatients) {
        set_.clear();
        keys_.assign(maxPatients, QueueKey{-1, 0, -1});
    }
    void push(int32_t id, int32_t prio, int64_t seq) {
        if (keys_[id].prio >= 0) throw std::logic_error("patient already queued");
        keys_[id] = {prio, seq, id};
        set_.insert(keys_[id]);
    }
    bool contains(int32_t id) const { return keys_[id].prio >= 0; }
    void updatePriority(int32_t id, int32_t prio) {
        if (!contains(id)) return;
        set_.erase(keys_[id]);
        keys_[id].prio = prio;
        set_.insert(keys_[id]);
    }
    int32_t popFront() {
        QueueKey k = *set_.begin();
        set_.erase(set_.begin());
        keys_[k.id].prio = -1;
        return k.id;
    }
    int32_t front() const { return set_.begin()->id; }
    bool empty() const { return set_.empty(); }
    size_t size() const { return set_.size(); }
    const std::set<QueueKey>& items() const { return set_; }

private:
    std::set<QueueKey> set_;
    std::vector<QueueKey> keys_;
};

}  // namespace hs
