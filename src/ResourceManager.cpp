#include "ResourceManager.h"

#include <algorithm>
#include <unordered_set>

namespace hs {

const char* resourceName(int r) {
    static const char* n[R_COUNT] = {"Doctors", "Triage nurses", "ER beds", "ICU beds", "Ward beds", "Lab slots"};
    return (r >= 0 && r < R_COUNT) ? n[r] : "?";
}

void ResourcePool::init(const std::string& name, int capacity) {
    name_ = name;
    occ_.assign(std::max(0, capacity), -1);
    free_ = {};
    for (int i = 0; i < (int)occ_.size(); ++i) free_.push(i);
    busy_ = 0;
    busyTime_ = 0;
}

int ResourcePool::acquire(int32_t patientId) {
    if (patientId < 0) throw std::logic_error("invalid patient id");
    if (free_.empty()) return -1;
    const int s = free_.top();
    free_.pop();
    if (occ_[s] != -1) throw std::logic_error(name_ + ": slot handed out twice");
    occ_[s] = patientId;
    ++busy_;
    return s;
}

void ResourcePool::release(int slot, int32_t patientId) {
    if (slot < 0 || slot >= (int)occ_.size()) throw std::logic_error(name_ + ": release of invalid slot");
    if (occ_[slot] != patientId) throw std::logic_error(name_ + ": release by non-occupant");
    occ_[slot] = -1;
    --busy_;
    free_.push(slot);
}

void ResourceManager::init(const Capacities& cap) {
    pools_[R_DOCTOR].init(resourceName(R_DOCTOR), cap.doctors);
    pools_[R_NURSE].init(resourceName(R_NURSE), cap.nurses);
    pools_[R_ER_BED].init(resourceName(R_ER_BED), cap.erBeds);
    pools_[R_ICU_BED].init(resourceName(R_ICU_BED), cap.icuBeds);
    pools_[R_WARD_BED].init(resourceName(R_WARD_BED), cap.wardBeds);
    pools_[R_LAB].init(resourceName(R_LAB), cap.labSlots);
}

bool ResourceManager::validate(std::string* err) const {
    for (int r = 0; r < R_COUNT; ++r) {
        const ResourcePool& p = pools_[r];
        std::unordered_set<int32_t> seen;
        int busy = 0;
        for (int s = 0; s < p.capacity(); ++s) {
            const int32_t o = p.occupant(s);
            if (o < 0) continue;
            ++busy;
            if (!seen.insert(o).second) {
                if (err) *err = p.name() + ": patient " + std::to_string(o) + " holds two slots";
                return false;
            }
        }
        if (busy != p.busy()) {
            if (err) *err = p.name() + ": busy counter mismatch";
            return false;
        }
    }
    return true;
}

}  // namespace hs
