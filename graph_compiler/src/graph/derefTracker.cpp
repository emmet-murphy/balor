#include "derefTracker.h"

namespace Balor {

void DerefTracker::makeNewDerefMap() { derefsToDerefNode = std::map<std::string, DerefNode *>(); }

DerefNode *DerefTracker::getDerefNode(SgBinaryOp *arrayIndex) {
    std::string derefString = arrayIndex->unparseToString();
    if (derefsToDerefNode.count(derefString)) {
        return derefsToDerefNode[derefString];
    }
    return nullptr;
}

void DerefTracker::saveDerefNode(SgBinaryOp *arrayIndex, DerefNode *deref) {
    derefsToDerefNode[arrayIndex->unparseToString()] = deref;
}

} // namespace Balor