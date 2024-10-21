#ifndef BALOR_AST_PARSER_H
#define BALOR_AST_PARSER_H

#include "../utility.h"
#include "graphGenerator.h"
#include "node.h"
#include "rose.h"
#include <queue>

namespace Balor {

class GraphGenerator;
class DerefTracker;
class PragmaParser;
class VariableMapper;
class LocalScalarNode;
class ControlFlowEdge;

class ReturnEdge;
class FunctionStartEdge;
class WriteNode;
class MergeStartEdge;
class ArithmeticUnitEdge;

class Node;
class DerefNode;
class FunctionCallNode;

class AstParser {
  public:
    AstParser(GraphGenerator *graphGenerator);

    GraphGenerator *graphGenerator;
    PragmaParser *pragmaParser;
    VariableMapper *variableMapper;
    DerefTracker *derefTracker;

    void parseAst(SgFunctionDefinition *topLevelFuncDef);

    void handleBB(std::vector<SgStatement *> statements);
    void catchBreakStatements();

    std::stack<std::queue<MergeStartEdge *>> breakMergeEdges;
    std::stack<bool> ifStatementBreaks;

    Node *handleFunctionCall(SgFunctionCallExp *funcCall);

    Node *readArithmeticExpression(SgExpression *expr);
    Node *processArithmeticExpression(VariantT variant, Node *lhs, Node *rhs);
    // Node *writeExpression(SgNode *lhs, Node *rhs);

    LocalScalarNode *getIteratorFromForInit(SgStatementPtrList statements);

    Node *readExpression(SgExpression *expr);
    Node *writeExpression(SgNode *lhs, Node *rhs);

    Node *addWrite(Node *variable, Node *rhs, Node *typeDependency);

    Node *handleIncOp(SgExpression *expr);


    void handleConditional(SgExpression *expr, Node *&branchNodeOut);
    void handleConditional(SgExpression *expr, Node *&branchNodeOut, Node *&comparisonNodeOut);

    void makeVariable(SgInitializedName *varDec);

    ReturnEdge *handleFuncDec(SgFunctionDeclaration *funcDec, Node *external);

    DerefNode *getDerefNode(SgBinaryOp *arrayIndex);
    DerefNode *getDerefNode(SgBinaryOp *arrayIndex, bool ignoreSaved);

    DerefNode *getDotNode(SgBinaryOp *dotExpr);

    FunctionStartEdge *getFunctionStartEdge(SgFunctionDeclaration *funcDec);
    ReturnEdge *getFunctionReturnEdge(SgFunctionDeclaration *funcDec);

    std::map<SgFunctionDeclaration *, FunctionStartEdge *> functionStartEdgeMap;
    std::map<SgFunctionDeclaration *, ReturnEdge *> functionReturnEdgeMap;

    std::queue<SgFunctionDeclaration *> functionDecsNeeded;
    std::set<SgFunctionDeclaration *> functionDecsComplete;

    std::map<SgFunctionDeclaration *, std::vector<FunctionCallNode *>> decsToCalls;

    std::set<SgInitializedName *> variableDeclarationsProcessed;

    Node *functionReturn = nullptr;
};
} // namespace Balor

#endif