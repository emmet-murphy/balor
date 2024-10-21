#ifndef BALOR_DEREF_TRACKER_H
#define BALOR_DEREF_TRACKER_H

#include "node.h"
#include "rose.h"

namespace Balor {

class DerefNode;
class ExpandableEdge;
class GraphGenerator;

class DerefTracker {
  public:
    DerefTracker() {}
    DerefNode *getDerefNode(SgBinaryOp *arrayIndex);
    void saveDerefNode(SgBinaryOp *arrayIndex, DerefNode *deref);

    void makeNewDerefMap();

  private:
    std::map<std::string, DerefNode *> derefsToDerefNode;
};

} // namespace Balor

#endif