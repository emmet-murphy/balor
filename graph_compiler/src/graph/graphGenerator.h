#ifndef BALOR_GRAPH_GENERATOR_H
#define BALOR_GRAPH_GENERATOR_H

#include "astParser.h"
#include "derefTracker.h"
#include "edge.h"
#include "node.h"
#include "pragmaParser.h"
#include "rose.h"
#include "variableMapper.h"
#include <memory>
#include <queue>
#include <unordered_set>

namespace Balor {

class VariableMapper;
class PragmaParser;
class DerefTracker;
class AstParser;

class Node;
class Edge;

class GraphGenerator {
  public:
    GraphGenerator(Sawyer::CommandLine::ParserResult parserResult);

    void generateGraph(SgFunctionDefinition *topLevelFuncDef);
    void printGraph();

    std::unique_ptr<VariableMapper> variableMapper;
    std::unique_ptr<PragmaParser> pragmaParser;
    std::unique_ptr<DerefTracker> derefTracker;
    std::unique_ptr<AstParser> astParser;

    std::vector<Node *> nodes;
    std::vector<std::unique_ptr<Node>> nodes_unq;

    std::vector<Edge *> edges;
    std::vector<std::unique_ptr<Edge>> edges_unq;

    bool checkArg(const std::string &arg);

    std::string getGroupName();
    void setGroupName(const std::string &groupName);

    int getBBID();
    void newBB();
    void newBB(bool onlyPrograml);

    int getFunctionID();
    void enterNewFunction();

    SgFunctionDeclaration *getFuncDec();
    void setFuncDec(SgFunctionDeclaration *funcDec);

    Node *stateNode = nullptr;

    void registerCalls(SgFunctionDeclaration *funcDec, int unrollFactor);

    void setFuncInlined(SgFunctionDeclaration *funcDec, bool inlined);
    bool getFuncInlined(SgFunctionDeclaration *funcDec);

    int getCallsNums(SgFunctionDeclaration *funcDec);
    int getCallSiteNums(SgFunctionDeclaration *funcDec);

    std::string datasetIndex;
    std::string graphType;

  private:
    // used to specify which function a node belongs to
    // for grouping on pdf
    std::string groupName;

    // to one-hot encodes bbID and functionID
    int bbID = 0;
    int functionID = 0;

    bool bbEmpty = true;

    SgFunctionDeclaration *funcDec = nullptr;

    std::map<SgFunctionDeclaration *, int> funcDecsToCallNums;
    std::map<SgFunctionDeclaration *, int> funcDecsToCallSiteNums;

    std::map<SgFunctionDeclaration *, bool> isDecInlined;


    std::map<std::string, bool> argMap;
};
} // namespace Balor

#endif