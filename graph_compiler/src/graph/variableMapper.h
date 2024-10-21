#ifndef BALOR_VARIABLE_MAPPER_H
#define BALOR_VARIABLE_MAPPER_H

#include "graphGenerator.h"
#include "node.h"

namespace Balor {
class GraphGenerator;

class Node;
class StructFieldNode;
class TypeStruct;

class VariableMapper {
  public:
    // VariableMapper() {}
    VariableMapper(GraphGenerator *graphGenerator) : graphGenerator(graphGenerator) {}

    struct Compare {
        using is_transparent = void; // important

        bool operator()(Balor::Node const *l, Balor::Node const *r) const { return l < r; }
    };

    GraphGenerator *graphGenerator;

    std::map<SgInitializedName *, Node *> variableToWriteNode;
    std::map<SgInitializedName *, Node *> variableToReadNode;

    std::set<Node *, Compare> nonReadVariables;
    std::map<SgInitializedName *, SgInitializedName *> underlyingVariableMap;

    std::map<std::string, std::string> resourceTypeMap;
    std::map<std::string, std::queue<std::tuple<std::string, int, int>>> arrayPartitionMap;

    std::map<SgType *, std::map<SgInitializedName *, StructFieldNode *>> structFieldMap;

    Node *createLocalVariableNode(SgInitializedName *variable);
    Node *createParameterVariableNode(SgInitializedName *variable);
    void setUnderlyingVariable(SgInitializedName *variable, SgInitializedName *underlyingVariable);
    void removeUnderlyingVariable(SgInitializedName *variable);
    void setParamNode(SgInitializedName *param, SgExpression *input);

    Node *readVariable(SgInitializedName *variable);
    Node *writeVariable(SgInitializedName *variable);

    SgInitializedName *getUnderlyingVariable(SgInitializedName *variable);

    void addArrayPragmas(const std::string &variableName, Node *pointerNode);

    void addStructTypeToMap(SgType *structType);
    StructFieldNode *getStructField(SgType *structType, SgInitializedName *variable);

    bool finishedMain = false;
};
} // namespace Balor

#endif