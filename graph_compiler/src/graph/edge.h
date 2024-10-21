#ifndef BALOR_EDGE_H
#define BALOR_EDGE_H

#include "graphGenerator.h"
#include "node.h"
#include "rose.h"
#include <queue>
namespace Balor {

class GraphGenerator;
class Node;

class Edge;

class TypeStruct;

class Edges {
  public:
    static GraphGenerator *graphGenerator;
    static Node *getPreviousControlFlowNode();
    static void updatePreviousControlFlowNode(Node *node);
    static void addPreviousControlFlowNodeChangeListener(Edge *edge);

    static void printSubControlFlowEdge(Node *source, Node *destination);
    static void printSubControlFlowEdge(Node *source, Node *destination, bool backEdge);
    static void printSubFunctionCallEdge(Node *source, Node *destination);
    static void printSubFunctionCallEdge(Node *source, Node *destination, int order);
    static void printSubMemoryAddressEdge(Node *source, Node *destination);
    static void printSubDataFlowEdge(Node *source, Node *destination);
    static void printSubDataFlowEdge(Node *source, Node *destination, int order);
    static void printPragmaEdge(Node *source, Node *destination, int order);

  private:
    static Node *previousControlFlowNode;
    static std::queue<Edge *> previousControlFlowNodeChangeListeners;
};

enum class EdgeVariant { DEFAULT, CONTROL_FLOW };

class Edge {
  public:
    Edge(Node *source, Node *destination);

    Node *source = nullptr;
    Node *destination = nullptr;

    int order = 0;

    virtual EdgeVariant getVariant() { return EdgeVariant::DEFAULT; }

    virtual std::string toString() = 0;
    virtual void run() = 0;
    virtual void runDeferred() {}
};

class ControlFlowEdge : public Edge {
  public:
    ControlFlowEdge(Node *destination) : Edge(nullptr, destination) { assert(destination); }

    EdgeVariant getVariant() override { return EdgeVariant::CONTROL_FLOW; }

    void run() override;
    std::string toString() override { return "Control Flow"; }
};

class UnrollPragmaEdge : public Edge {
  public:
    UnrollPragmaEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Unroll Pragma Edge"; }
};

class PipelinePragmaEdge : public Edge {
  public:
    PipelinePragmaEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Pipeline Pragma Edge"; }
};

class ArrayPartitionPragmaEdge : public Edge {
  public:
    ArrayPartitionPragmaEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Array Partition Edge"; }
};

class ResourceAllocationPragmaEdge : public Edge {
  public:
    ResourceAllocationPragmaEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Resource Allocation Edge"; }
};

class InlineFunctionPragmaEdge : public Edge {
  public:
    InlineFunctionPragmaEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Inline Function Edge"; }
};

class DataFlowEdge : public Edge {
  public:
    DataFlowEdge(Node *source, Node *destination) : Edge(source, destination) {}
    DataFlowEdge(Node *source, Node *destination, int order) : Edge(source, destination) { this->order = order; }

    void run() override;

    std::string toString() override { return "Data Flow"; }
};

class MemoryAddressEdge : public Edge {
  public:
    MemoryAddressEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    Node *typeDependency = nullptr;

    TypeStruct getElemType();

    std::string toString() override { return "Memory Address"; }
};

class SpecifyAddressEdge : public Edge {
  public:
    SpecifyAddressEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;

    std::string toString() override { return "Specify Address Edge"; }
};
class StructFieldNode;

class WriteMemoryElementEdge : public Edge {
  public:
    WriteMemoryElementEdge(Node *source, Node *destination);

    void run() override;
    std::string toString() override { return "Write Memory Element Edge"; }
};

class ReadMemoryElementEdge : public Edge {
  public:
    ReadMemoryElementEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Read Memory Element Edge"; }
};

class ProgramlBranchEdge : public Edge {
  public:
    ProgramlBranchEdge() : Edge(nullptr, nullptr) {}

    void run() override;
    std::string toString() override { return "Programl Branch Edge"; }
};

// A merge merges the control flow
// from two different operations to one
// Add this edge after the first of the two operations
class MergeStartEdge : public Edge {
  public:
    MergeStartEdge() : Edge(nullptr, nullptr) {}

    Node *source1 = nullptr;
    // When this edge is ran, it doesn't print anything
    // As it doesn't know where to merge to yet
    // So it just saves where to merge from
    void run() override;
    std::string toString() override { return "Merge Open Edge"; }
};

// A merge merges the control flow
// from two different operations to one
// Add this edge after the second of the two operations
class MergeEndEdge : public Edge {
  public:
    // you need to pass the first edge when making the second
    MergeEndEdge(MergeStartEdge *openEdge) : openEdge(openEdge), Edge(nullptr, nullptr) {}

    MergeStartEdge *openEdge;
    Node *source2 = nullptr;

    // The first run still doesn't know where to merge to
    // So it stores the second place to merge from in source2
    // and tells the Edges class to call runDeferred after
    // a control flow edge is added from source2
    void run() override;

    // After a control flow edge is added from source2
    // we can get the current control flow node
    // and add an edge from source1 to it
    void runDeferred() override;
    std::string toString() override { return "Merge Close Edge"; }
};

class SextDataFlowEdge : public Edge {
  public:
    SextDataFlowEdge(Node *source, Node *destination) : Edge(source, destination) {}

    SextDataFlowEdge(Node *source, Node *destination, int newWidth) : newWidth(newWidth), Edge(source, destination) {
        widthOverridden = true;
    }

    bool widthOverridden = false;
    int newWidth;

    void run() override;
    std::string toString() override { return "Sext Edge"; }
};

class ParameterLoadDataFlowEdge : public Edge {
  public:
    ParameterLoadDataFlowEdge(Node *source, Node *destination) : Edge(source, destination) {}

    void run() override;
    std::string toString() override { return "Parameter Load Edge"; }
};

class ParameterInitializeEdge : public Edge {
  public:
    ParameterInitializeEdge(Node *parameter) : Edge(parameter, nullptr) {}

    void run() override;
    std::string toString() override { return "Parameter Initialize Edge"; }
};

class VariableDeclareEdge : public Edge {
  public:
    VariableDeclareEdge(Node *parameter) : Edge(parameter, nullptr) {}

    void run() override;
    std::string toString() override { return "Variable Declare Edge"; }
};

class ReturnEdge : public Edge {
  public:
    ReturnEdge(Node *functionReturn, SgFunctionDeclaration *funcDec)
        : Edge(nullptr, nullptr), functionReturn(functionReturn), funcDec(funcDec) {}

    ReturnEdge() : Edge(nullptr, nullptr) {}

    std::vector<Node *> returnLocations;
    virtual void setNodeVariables(Node *node);

    Node *functionReturn = nullptr;
    SgFunctionDeclaration *funcDec = nullptr;

    virtual Node *getNode();
    void run() override;
    std::string toString() override { return "Return Edge"; }

  protected:
    Node *privateNode = nullptr;
};

class UndefinedFunctionEdge : public ReturnEdge {
  public:
    UndefinedFunctionEdge(const std::string &functionName, int funcID)
        : functionName(functionName), funcID(funcID), ReturnEdge() {}

    std::string functionName;
    int funcID;

    void setNodeVariables(Node *node) override;
    Node *getNode() override;
    std::string toString() override { return "Undefined Function Edge"; }

  private:
    Node *privateReturnNode = nullptr;
};

class PreLoopEdge : public Edge {
  public:
    PreLoopEdge() : Edge(nullptr, nullptr) {}

    Node *loopConditionStart = nullptr;

    void run() override { Edges::addPreviousControlFlowNodeChangeListener(this); }
    void runDeferred() override { loopConditionStart = Edges::getPreviousControlFlowNode(); }
    std::string toString() override { return "Pre Loop Edge"; }
};

class LoopBackEdge : public Edge {
  public:
    LoopBackEdge(PreLoopEdge *preLoopEdge, Node *branch)
        : Edge(nullptr, nullptr), preLoopEdge(preLoopEdge), branch(branch) {}

    Node *branch;
    PreLoopEdge *preLoopEdge;
    void run() override;
    std::string toString() override { return "Loop Back Edge"; }
};

class RevertControlFlowEdge : public Edge {
  public:
    RevertControlFlowEdge(Node *nodeToRevertTo) : Edge(nullptr, nullptr), nodeToRevertTo(nodeToRevertTo) {}

    Node *nodeToRevertTo;

    void run() override { Edges::updatePreviousControlFlowNode(nodeToRevertTo); }
    std::string toString() override { return "Revert Control Flow Edge"; }
};

class FunctionStartEdge : public Edge {
  public:
    FunctionStartEdge(Node *external) : external(external), Edge(nullptr, nullptr) {}

    Node *external;

    std::vector<Node *> callLocations;
    std::string functionName;

    void run() override;
    void runDeferred() override;

    std::string toString() override { return "Function Start Edge"; }
};

class FunctionCallEdge : public Edge {
  public:
    FunctionCallEdge(Node *funcCallNode, SgFunctionDeclaration *funcDec)
        : funcCallNode(funcCallNode), funcDec(funcDec), Edge(nullptr, nullptr) {}

    SgFunctionDeclaration *funcDec;
    std::vector<SgExpression *> parameters;
    Node *funcCallNode;

    void run() override;

    std::string toString() override { return "Function Call Node Edge"; }
};

class BackControlFlowEdge : public Edge {
  public:
    BackControlFlowEdge(Node *destination) : Edge(nullptr, destination) { assert(destination); }

    void run() override;
    std::string toString() override { return "Back Control Flow"; }
};

class ArithmeticUnitEdge : public Edge {
  public:
    ArithmeticUnitEdge(Node *lhs, Node *rhs, Node *unit) : lhs(lhs), rhs(rhs), unit(unit), Edge(nullptr, nullptr) {}

    Node *lhs, *rhs, *unit;

    void run() override;
    std::string toString() override { return "Arithmetic Unit"; }

    TypeStruct getType();
};

class ImplicitCastDataFlowEdge : public Edge {
  public:
    ImplicitCastDataFlowEdge(Node *lhs, Node *rhs) : Edge(lhs, rhs), typeDependency(rhs) {}

    ImplicitCastDataFlowEdge(Node *lhs, Node *rhs, Node *typeDependency)
      : Edge(lhs, rhs), typeDependency(typeDependency) {}

    Node *typeDependency;

    std::string toString() { return "Write Dataflow"; }

    void run() override;
};

class StructAccessEdge : public Edge {
  public:
    StructAccessEdge(Node *accessNode) : accessNode(accessNode), Edge(nullptr, nullptr) {}

    Node *accessNode;


    std::string toString() { return "Struct Access"; }

    void run() override;

};

} // namespace Balor
#endif