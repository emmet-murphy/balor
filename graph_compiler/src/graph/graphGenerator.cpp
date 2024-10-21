#include "graphGenerator.h"
#include "../utility.h"
#include "args.h"
#include "nodeUtils.h"
#include "rose.h"
#include <Rose/CommandLine.h>
#include <boost/algorithm/string.hpp>
#include <cassert>

namespace Balor {

GraphGenerator::GraphGenerator(Sawyer::CommandLine::ParserResult parserResult) {
    for (auto arg : ARGS) {
        std::string argName = arg.first;
        if (parserResult.have(argName)) {
            argMap[argName] = true;
        } else {
            argMap[argName] = false;
        }
    }

    datasetIndex = parserResult.parsed("datasetIndex").back().asString();
    graphType = parserResult.parsed("graphType").back().asString();

    variableMapper = std::make_unique<VariableMapper>(this);
    pragmaParser = std::make_unique<PragmaParser>(this);
    derefTracker = std::make_unique<DerefTracker>();
    astParser = std::make_unique<AstParser>(this);
}

bool GraphGenerator::checkArg(const std::string &arg) {
    assert(argMap.count(arg));
    return argMap[arg];
}

// Print a dot file description of the graph to the terminal
void GraphGenerator::printGraph() {
    std::cerr << "printing graph" << std::endl;
    // make a directed graph
    std::cout << "digraph {" << std::endl;
    std::cout << "newrank=\"true\";" << std::endl;

    std::vector<Node *> nodesFrozen = nodes;

    // node ID starts at 0
    Nodes::resetNodeID();
    // for each node
    for (Node *node : nodesFrozen) {
        std::cerr << "node" << std::endl;

        node->print();
    }
    std::cerr << "nodes done" << std::endl;
    std::vector<Edge *> edgesFrozen = edges;
    for (Edge *edge : edgesFrozen) {
        std::cerr << edge->toString() << std::endl;
        std::cerr << "edge" << std::endl;

        edge->run();
    }
    // close the directed graph
    std::cout << "}" << std::endl;
}

std::string GraphGenerator::getGroupName() {
    if (stateNode) {
        return stateNode->groupName;
    }
    return groupName;
}
void GraphGenerator::setGroupName(const std::string &groupName) { this->groupName = groupName; }

int GraphGenerator::getBBID() {
    if (stateNode) {
        return stateNode->bbID;
    }
    bbEmpty = false;
    return bbID;
}
void GraphGenerator::newBB() {
    newBB(false);
}

void GraphGenerator::newBB(bool onlyPrograml) {
    if(onlyPrograml && !checkArg(PROXY_PROGRAML)){
        return;
    }
    if (!bbEmpty) {
        bbID++;
    }
    bbEmpty = true;
}

int GraphGenerator::getFunctionID() {
    if (stateNode) {
        return stateNode->functionID;
    }
    return functionID;
}
void GraphGenerator::enterNewFunction() { functionID++; }

SgFunctionDeclaration *GraphGenerator::getFuncDec(){
    if(stateNode){
        return stateNode->funcDec;
    }
    assert(funcDec);
    return funcDec;
}

void GraphGenerator::setFuncDec(SgFunctionDeclaration *funcDec){
    this->funcDec = funcDec;
}

void GraphGenerator::registerCalls(SgFunctionDeclaration *funcDec, int unrollFactor){
    if(!funcDecsToCallNums.count(funcDec)){
        funcDecsToCallNums[funcDec] = unrollFactor;
        funcDecsToCallSiteNums[funcDec] = 1;
    } else {
        funcDecsToCallNums[funcDec] += unrollFactor;
        funcDecsToCallSiteNums[funcDec]++;
    }
}

void GraphGenerator::setFuncInlined(SgFunctionDeclaration *funcDec, bool inlined){
    isDecInlined[funcDec] = inlined;
}

bool GraphGenerator::getFuncInlined(SgFunctionDeclaration *funcDec){
    assert(isDecInlined.count(funcDec));
    return isDecInlined[funcDec];
}

int GraphGenerator::getCallsNums(SgFunctionDeclaration *funcDec){
    assert(funcDecsToCallNums.count(funcDec));
    return funcDecsToCallNums[funcDec];
}
int GraphGenerator::getCallSiteNums(SgFunctionDeclaration *funcDec){
    assert(funcDecsToCallSiteNums.count(funcDec));
    return funcDecsToCallSiteNums[funcDec];
}

void GraphGenerator::generateGraph(SgFunctionDefinition *topLevelFuncDef) {
    Edges::graphGenerator = this;
    Nodes::graphGenerator = this;
    astParser->parseAst(topLevelFuncDef);
}

} // namespace Balor
