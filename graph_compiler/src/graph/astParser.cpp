#include "astParser.h"
#include "args.h"
#include "nodeUtils.h"
#include <cassert>

namespace Balor {

AstParser::AstParser(GraphGenerator *graphGenerator) : graphGenerator(graphGenerator) {
    pragmaParser = graphGenerator->pragmaParser.get();
    variableMapper = graphGenerator->variableMapper.get();
    derefTracker = graphGenerator->derefTracker.get();
}

void AstParser::makeVariable(SgInitializedName *varDec) {
    if (variableDeclarationsProcessed.count(varDec)) {
        return;
    }

    variableDeclarationsProcessed.insert(varDec);

    // add the variable to the graph
    // as local variable can consume resources
    Node *localVariable = variableMapper->createLocalVariableNode(varDec);
    new VariableDeclareEdge(localVariable);
}

void AstParser::catchBreakStatements() {
    std::queue<MergeStartEdge *> currentBreakMergeEdges = breakMergeEdges.top();
    breakMergeEdges.pop();
    while (!currentBreakMergeEdges.empty()) {
        MergeStartEdge *edge = currentBreakMergeEdges.front();
        currentBreakMergeEdges.pop();
        new MergeEndEdge(edge);
    }
}

// Handle every line of code in a basic block
void AstParser::handleBB(std::vector<SgStatement *> statements) {

    // we only reuse array dereferences inside the same BB
    derefTracker->makeNewDerefMap();

    // the code currently can handle a maximum of 1 return statement
    // per function
    int numberOfReturnStatements = 0;

    // for each line of code in a basic block
    for (SgStatement *statement : statements) {
        if (SgPragmaDeclaration *pragmaDec = isSgPragmaDeclaration(statement)) {
            continue;
        }

        // if the line of code is 1 or more variable declarations
        if (SgVariableDeclaration *varDecStatement = isSgVariableDeclaration(statement)) {
            // for each variable declared in the line of code
            for (SgInitializedName *varDec : varDecStatement->get_variables()) {
                makeVariable(varDec);
                // if the variable is initialized
                if (SgExpression *expr = isSgExpression(varDec->get_initptr())) {
                    // if it is initialized with an expression
                    if (SgAssignInitializer *init = isSgAssignInitializer(expr)) {
                        Node *var = variableMapper->readVariable(varDec);
                        Node *rhs = readExpression(init->get_operand());
                        Node *writeNode = writeExpression(varDec, rhs);
                        new ControlFlowEdge(writeNode);
                    } else if (SgConstructorInitializer *init = isSgConstructorInitializer(expr)) {
                        std::cout << "Constructors are currently excluded" << std::endl;
                    } else if (auto *init = isSgAggregateInitializer(expr)) {
                        Node *variable = variableMapper->readVariable(varDec);
                        if(graphGenerator->checkArg(PROXY_PROGRAML)){
                            Node *bitcast = new BitcastNode();
                            new ControlFlowEdge(bitcast);
                            new DataFlowEdge(variable, bitcast);

                            // make empty func dec, the signature doesn't matter
                            SgType *returnType = SageBuilder::buildIntType();
                            SgFunctionParameterList *paramList = SageBuilder::buildFunctionParameterList();

                            // Create a function declaration
                            SgName functionName("memcopy");
                            SgFunctionDeclaration *funcDec =
                                SageBuilder::buildNondefiningFunctionDeclaration(functionName, returnType, paramList, NULL);

                            functionDecsNeeded.push(funcDec);
                            FunctionCallNode *funcCallNode = new FunctionCallNode();
                            FunctionCallEdge *funcCallEdge = new FunctionCallEdge(funcCallNode, funcDec);

                            new DataFlowEdge(bitcast, funcCallNode);

                            TypeStruct boolType = TypeStruct(DataType::INTEGER, 1);
                            Node *constant1 = new FakeConstantNode("Memcopy Bool", boolType);
                            new DataFlowEdge(constant1, funcCallNode);

                            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                            Node *constant2 = new FakeConstantNode("Memcopy Constant Pointer", pointerType);
                            new DataFlowEdge(constant2, funcCallNode);

                            Node *constant3 = new FakeConstantNode("Memcopy Input ?", pointerType);
                            new DataFlowEdge(constant3, funcCallNode);
                        } else {
                            Node *var = variableMapper->readVariable(varDec);
                            Node *rhs = new FakeConstantNode("Array Initialization", var->getType());
                            Node *writeNode = writeExpression(varDec, rhs);
                            int numVals = init->get_initializers()->get_expressions().size();
                            writeNode->unrollFactor.full *= numVals;
                            new ControlFlowEdge(writeNode);
                        }

                    } else {
                        throw std::runtime_error("variable initialized in unknown way: " + expr->unparseToString());
                    }
                }
            }

            // if the line of code is an expression
        } else if (SgExprStatement *exprStatement = isSgExprStatement(statement)) {
            // read the expression
            readExpression(exprStatement->get_expression());

            // if its a return statement
        } else if (SgReturnStmt *returnStatement = isSgReturnStmt(statement)) {
            // record that a return statement has been found
            numberOfReturnStatements++;

            // process the expression
            // and if it returns a value, save it in the member variable
            // so whatever called handleStatements can find it
            functionReturn = readExpression(returnStatement->get_expression());

            // if its a for statement
        } else if (SgForStatement *forStatement = isSgForStatement(statement)) {
            // get the init statement
            SgStatementPtrList forInit = forStatement->get_for_init_stmt()->get_init_stmt();

            LocalScalarNode *iterator = getIteratorFromForInit(forInit);
            if(!iterator){
                iterator = new LocalScalarNode("fake iterator: this should be fixed");
            }
            iterator->inIteratorBoundsRegion = true;

            // handle the init statement
            graphGenerator->newBB(true);
            handleBB(forInit);
            iterator->inIteratorBoundsRegion = false;

            new ProgramlBranchEdge();

            breakMergeEdges.push(std::queue<MergeStartEdge *>());

            // safely cast to the body to a bb
            SgBasicBlock *bb = isSgBasicBlock(forStatement->get_loop_body());
            if (!bb) {
                bb = SageBuilder::buildBasicBlock(forStatement->get_loop_body());
            }
            assert(bb);

            // get any pragmas in this bb and apply them
            pragmaParser->parsePragmas(bb);

            // unroll pragmas in a bb don't affect the condition
            // other pragmas (pipeline) do
            pragmaParser->enterLoopCondition();

            // store the next instruction that is executed
            // so that we can get back to it
            // when we want to re-execute the condition
            PreLoopEdge *preLoopEdge = new PreLoopEdge();

            graphGenerator->newBB();
            derefTracker->makeNewDerefMap();

            // we need the branch making control flow edges
            // and the comparison for adding pragma nodes to
            // and they're output by reference, so here's where
            // we store them
            Node *branchNode = nullptr;
            Node *comparisonNode = nullptr;
            handleConditional(forStatement->get_test_expr(), branchNode, comparisonNode);

            iterator->processComparison(comparisonNode);

            // finished handling the condition, so we can
            // mark any nodes below this point with the unroll factor
            pragmaParser->exitLoopCondition();

            if (pragmaParser->getUnrollFactor().first > 1) {
                PragmaNode *pragma = new UnrollPragmaNode(pragmaParser->getUnrollFactor().first);
                new UnrollPragmaEdge(pragma, comparisonNode);
            }

            if (pragmaParser->getPipelined()) {
                PragmaNode *pragma = new PipelinePragmaNode();
                new PipelinePragmaEdge(pragma, comparisonNode);
            }

            graphGenerator->newBB();
            // handle the body of the for loop
            handleBB(bb->getStatementList());

            new ProgramlBranchEdge();

            if (forStatement->get_increment()->variantT() != V_SgNullExpression) {
                // unroll pragmas don't affect the loop increment
                pragmaParser->enterLoopInc();

                iterator->inIncrementRegion = true;

                graphGenerator->newBB(true);
                // handle the increment expression
                readExpression(forStatement->get_increment());

                iterator->inIncrementRegion = false;

                new ProgramlBranchEdge();

                pragmaParser->exitLoopInc();
            }

            // loop back to the first node we need to execute
            // to evaluate the condition,
            // and then mark the branch node as the predecessor
            new LoopBackEdge(preLoopEdge, branchNode);

            // unapply any pragmas from this bb
            pragmaParser->unstackPragmas();

            catchBreakStatements();

            graphGenerator->newBB();
            derefTracker->makeNewDerefMap();

            // if its a while statement
        } else if (SgWhileStmt *whileStmt = isSgWhileStmt(statement)) {
            // we need the branch making control flow edges
            // and the comparison for adding pragma nodes to
            // and they're output by reference, so here's where
            // we store them
            Node *branchNode = nullptr;
            Node *comparisonNode = nullptr;

            new ProgramlBranchEdge();

            // store the next instruction that is executed
            // so that we can get back to it
            // when we want to re-execute the condition
            PreLoopEdge *preLoopEdge = new PreLoopEdge();

            graphGenerator->newBB();
            SgStatement *condStatement = whileStmt->get_condition();
            SgExprStatement *condExprStatement = isSgExprStatement(condStatement);
            assert(condExprStatement);
            SgExpression *condExpr = condExprStatement->get_expression();
            handleConditional(condExpr, branchNode, comparisonNode);

            breakMergeEdges.push(std::queue<MergeStartEdge *>());

            // safely cast to the body to a bb
            SgBasicBlock *bb = isSgBasicBlock(whileStmt->get_body());
            if (!bb) {
                bb = SageBuilder::buildBasicBlock(whileStmt->get_body());
            }
            assert(bb);

            // get any pragmas in this bb and apply them
            pragmaParser->parsePragmas(bb);

            if (pragmaParser->getUnrollFactor().first > 1) {
                PragmaNode *pragma = new UnrollPragmaNode(pragmaParser->getUnrollFactor().first);
                new UnrollPragmaEdge(pragma, comparisonNode);            
            }

            graphGenerator->newBB();
            // handle the body of the for loop
            handleBB(bb->getStatementList());

            new ProgramlBranchEdge();

            // loop back to the first node we need to execute
            // to evaluate the condition,
            // and then mark the branch node as the predecessor
            new LoopBackEdge(preLoopEdge, branchNode);

            // unapply any pragmas from this bb
            pragmaParser->unstackPragmas();

            catchBreakStatements();
            derefTracker->makeNewDerefMap();

        } else if (SgDoWhileStmt *doWhileStmt = isSgDoWhileStmt(statement)) {
            throw std::runtime_error("Do While loops not currently supported");
        } else if (SgIfStmt *ifStmt = isSgIfStmt(statement)) {
            // merging control flow requires two edges
            // have to pass the first to the second
            // so its declared here
            MergeStartEdge *trueBodyMergeStartEdge;

            // branch node is an output by reference
            // this is where we store it
            Node *branchNode;

            // safely cast condition to an expression statement
            if (SgExprStatement *exprStatement = isSgExprStatement(ifStmt->get_conditional())) {
                // handle if statement condition
                handleConditional(exprStatement->get_expression(), branchNode);
            } else {
                throw std::runtime_error("Condition of an if statement wasn't an expression statement: " +
                                         ifStmt->get_conditional()->unparseToString());
            }
            // safely cast if body to a bb
            graphGenerator->newBB();

            ifStatementBreaks.push(false);
            if (SgBasicBlock *bb = isSgBasicBlock(ifStmt->get_true_body())) {
                // handle if body
                handleBB(bb->getStatementList());
            } else {
                std::vector<SgStatement *> statements;
                statements.push_back(ifStmt->get_true_body());
                handleBB(statements);
            }

            bool trueBodyBroke = ifStatementBreaks.top();
            ifStatementBreaks.pop();

            // if the if statement didn't break
            if (!trueBodyBroke) {
                new ProgramlBranchEdge();
                // mark that the previous node will need to merge
                trueBodyMergeStartEdge = new MergeStartEdge();
            }

            bool falseBodyBroke = false;
            // revert control flow predecessor to branch node
            new RevertControlFlowEdge(branchNode);

            if (ifStmt->get_false_body()) {
                ifStatementBreaks.push(false);

                // handle else body
                graphGenerator->newBB();
                // safely cast else body to a bb
                if (SgBasicBlock *bb = isSgBasicBlock(ifStmt->get_false_body())) {
                    handleBB(bb->getStatementList());
                } else {
                    std::vector<SgStatement *> statements;
                    statements.push_back(ifStmt->get_false_body());
                    handleBB(statements);
                }

                falseBodyBroke = ifStatementBreaks.top();
                ifStatementBreaks.pop();

                // if the if statement didn't break
                if (!falseBodyBroke) {
                    new ProgramlBranchEdge();
                }
            }

            // if both broke, don't look at rest of bb statements
            if (falseBodyBroke && trueBodyBroke) {
                break;
            } else if (!trueBodyBroke) {
                // mark that the next node to execute
                // has to merge the node marked in trueBodyMergeStartEdge
                new MergeEndEdge(trueBodyMergeStartEdge);
            }

        } else if (auto breakStatement = isSgBreakStmt(statement)) {
            new ProgramlBranchEdge();
            if (ifStatementBreaks.empty()) {
                std::runtime_error("Found a break statement not inside an if statement");
            }
            ifStatementBreaks.top() = true;
            breakMergeEdges.top().push(new MergeStartEdge());
        } else if (SgBasicBlock *bb = isSgBasicBlock(statement)){
            pragmaParser->parsePragmas(bb);
            handleBB(bb->getStatementList());
        } else {
            throw std::runtime_error("Unsupported top level statement found: " + statement->unparseToString());
        }
    }

    if (numberOfReturnStatements > 1) {
        throw std::runtime_error("A maximum of one return statement per function is supported");
    }
}

Node *AstParser::handleFunctionCall(SgFunctionCallExp *funcCall) {
    // get the function declaration
    SgFunctionDeclaration *funcDec = Balor::getFuncDecFromCall(funcCall);
    if (graphGenerator->checkArg(INLINE_FUNCTIONS) && pragmaParser->parseInlinePragma(funcDec)) {
        SgFunctionDeclaration *currentFuncDec = graphGenerator->getFuncDec();
        graphGenerator->registerCalls(funcDec, pragmaParser->getUnrollFactor().full);
        functionDecsComplete.insert(funcDec);
        graphGenerator->setFuncDec(funcDec);
        // start parameter index at 0
        int paramIndex = 0;

        // foreach parameter in the function call
        for (SgExpression *argExpr : funcCall->get_args()->get_expressions()) {
            // get the matching parameter in the function declaration
            // using the param index
            SgInitializedName *varDec = funcDec->get_args()[paramIndex];

            // and link the parameter variable to
            // whatever expression was passed to that parameter
            // could be a constant, a variable, another function call
            variableMapper->setParamNode(varDec, argExpr);

            // and update the param index
            paramIndex++;
        }

        graphGenerator->newBB();
        // process each line of the function body
        handleBB(funcDec->get_definition()->get_body()->getStatementList());

        graphGenerator->setFuncDec(currentFuncDec);
        // when inlined, dataflow edges come from the actual node
        return functionReturn;
    } else {

        FunctionCallNode *funcCallNode = new FunctionCallNode();
        FunctionCallEdge *funcCallEdge = new FunctionCallEdge(funcCallNode, funcDec);
        functionDecsNeeded.push(funcDec);

        if (!decsToCalls.count(funcDec)) {
            decsToCalls[funcDec] = std::vector<FunctionCallNode *>();
        }
        decsToCalls[funcDec].push_back(funcCallNode);

        for (SgExpression *argExpr : funcCall->get_args()->get_expressions()) {
            funcCallEdge->parameters.push_back(argExpr);
        }

        funcCallNode->setType(funcDec->get_orig_return_type()->findBaseType()->unparseToString());

        // when not inlined, dataflow edges come from the call node
        return funcCallNode;
    }
}

// Handle an expression
// Expressions are deeply nested, with each expression having other expressions
// as operands it is dependant on.
Node *AstParser::readExpression(SgExpression *expr) {
    // if the expression is an assignment operator
    if (SgAssignOp *assignOp = isSgAssignOp(expr)) {
        // handle the rhs of the assignment
        Node *rhs = readExpression(assignOp->get_rhs_operand());
        // and write to the lhs of the assignment
        Node *writeNode = writeExpression(assignOp->get_lhs_operand(), rhs);

        new ControlFlowEdge(writeNode);

        return rhs;
        // if the expression is a variable reference
    } else if (SgVarRefExp *varRef = isSgVarRefExp(expr)) {
        // get the actual variable
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

        // get the pointer/memory element
        Node *variableNode = variableMapper->readVariable(varDec);

        // reading from some variables doesn't add a read node
        if (variableMapper->nonReadVariables.count(variableNode) > 0) {
            return variableNode;
        }

        // if its actually a variable, you need a read node
        Node *read = new ReadNode(variableNode);

        new ControlFlowEdge(read);

        // scalar variables need an address edge from an alloca
        // if mem elements, all reads need an element edge
        new ReadMemoryElementEdge(variableNode, read);

        return read;

        // if its an array indexing
    } else if (SgPntrArrRefExp *arrayIndex = isSgPntrArrRefExp(expr)) {

        // see if derefence has happened before in this BB
        // if yes, returns the already existing deref node
        // if not, it will add all the nodes for derefencing to the graph
        // and return the deref node
        DerefNode *deref = getDerefNode(arrayIndex);

        Node *memoryElement = deref->memoryElement;


        Node *readDataNode = new ReadNode(deref->baseTypeDependency);

        new DataFlowEdge(deref, readDataNode);
        new ControlFlowEdge(readDataNode);

        // prints a memory address edge if not proxying programl
        // otherwise prints nothing
        new ReadMemoryElementEdge(memoryElement, readDataNode);

        // the read data node provides the out-going data
        // from all of this
        return readDataNode;

    } else if (SgPointerDerefExp *pointerDeref = isSgPointerDerefExp(expr)) {
        SgVarRefExp *varRef = isSgVarRefExp(pointerDeref->get_operand());
        assert(varRef);
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
        Node *variable = variableMapper->readVariable(varDec);

        Node *readNode = new ReadNode(variable);
        new ParameterLoadDataFlowEdge(variable, readNode);

        new ControlFlowEdge(readNode);

        new ReadMemoryElementEdge(variable, readNode);

        return readNode;

    } else if (Utils::isComparisonOp(expr->variantT())) {
        SgBinaryOp *compOp = isSgBinaryOp(expr);
        Node *rhs = readExpression(compOp->get_rhs_operand());
        Node *lhs = readExpression(compOp->get_lhs_operand());

        ComparisonNode *comparisonNode = new ComparisonNode();

        comparisonNode->lhs = lhs;
        comparisonNode->rhs = rhs;

        ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, comparisonNode);
        comparisonNode->edge = edge;

        return comparisonNode;
    } else if (SgNotOp *notOp = isSgNotOp(expr)) {
        Node *input = readExpression(notOp->get_operand());

        Node *notNode = new UnaryOpNode(input, "Not");
        new DataFlowEdge(input, notNode);

        return notNode;
        // if its a +=, -=, *= or /=
    } else if (Utils::isUpdateOp(expr->variantT())) {
        // add the correct arithmetic node
        Node *arithmeticNode = readArithmeticExpression(expr);

        // cast to a binary operation to get access to member variables
        SgBinaryOp *binaryOp = isSgBinaryOp(expr);

        // and write to the lhs of the assignment
        Node *writeNode = writeExpression(binaryOp->get_lhs_operand(), arithmeticNode);
        new ControlFlowEdge(writeNode);
        return writeNode;
        // if its an addition, subtraction, multiplication or division
    } else if (Utils::isBinaryArithmeticNode(expr->variantT())) {
        // Lots of operations consist partially of arithmetic nodes
        // so its in a function for reuse
        Node *arithmeticNode = readArithmeticExpression(expr);
        return arithmeticNode;
        // if its a function call
    } else if (SgFunctionCallExp *funcCall = isSgFunctionCallExp(expr)) {
        // function calls can be statements or expressions
        // so the same function is called from both places
        return handleFunctionCall(funcCall);
        // if its a ++ or --
    } else if (Utils::isIncOrDecOp(expr->variantT())) {
        return handleIncOp(expr);
        // if its an integer constant
    } else if (SgIntVal *intVal = isSgIntVal(expr)) {
        TypeStruct intType = TypeStruct(DataType::INTEGER, 32);
        // add it to the graph
        Node *constant = new ConstantNode(intVal->get_value(), intType);
        return constant;
    } else if (SgLongLongIntVal *intVal = isSgLongLongIntVal(expr)) {
        // add it to the graph
        TypeStruct longIntType = TypeStruct(DataType::INTEGER, 64);
        Node *constant = new ConstantNode(intVal->get_value(), longIntType);
        return constant;
        // if its a boolean constant
    } else if (SgBoolValExp *boolVal = isSgBoolValExp(expr)) {
        TypeStruct boolType = TypeStruct(DataType::INTEGER, 1);
        Node *constant = new FakeConstantNode("Bool: " + std::to_string(boolVal->get_value()), boolType);
        return constant;
    } else if (SgDoubleVal *doubleVal = isSgDoubleVal(expr)) {
        TypeStruct doubleType = TypeStruct(DataType::FLOAT, 64);
        Node *constant = new ConstantNode(doubleVal->get_value(), doubleType);
        return constant;
    } else if(SgLongIntVal *longIntVal = isSgLongIntVal(expr)){
        TypeStruct longType = TypeStruct(DataType::INTEGER, 64);
        Node *constant = new ConstantNode(longIntVal->get_value(), longType);
        return constant;
    } else if(SgUnsignedLongVal *unsignedLongVal = isSgUnsignedLongVal(expr)){
        TypeStruct longType = TypeStruct(DataType::INTEGER, 64);
        longType.isUnsigned = true;
        Node *constant = new ConstantNode(unsignedLongVal->get_value(), longType);
        return constant;
    } else if (SgCastExp *castExpr = isSgCastExp(expr)) {
        std::string type = castExpr->get_type()->findBaseType()->unparseToString();

        Node *input = readExpression(castExpr->get_operand());

        if (input->getVariant() == NodeVariant::CONSTANT) {
            input->setType(type);
            return input;
        } else {
            return input;
        }
    } else if (expr->variantT() == V_SgDotExp || expr->variantT() == V_SgArrowExp) {
        SgBinaryOp *binaryOp = isSgBinaryOp(expr);
        assert(binaryOp);

        if(graphGenerator->checkArg(PROXY_PROGRAML)){
            DerefNode *dotDeref = getDotNode(binaryOp);

            if (binaryOp->get_rhs_operand()->get_type()->variantT() == V_SgArrayType) {
                Node *conversionDeref = new DerefNode();
                TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                conversionDeref->setType(pointerType);
                new ControlFlowEdge(conversionDeref);
                new DataFlowEdge(dotDeref, conversionDeref);

                Node *constantA = new FakeConstantNode("0", pointerType);
                Node *constantB = new FakeConstantNode("0", pointerType);

                new DataFlowEdge(constantA, conversionDeref);
                new DataFlowEdge(constantB, conversionDeref);

                return conversionDeref;
            } else {
                Node *read = new ReadNode(dotDeref->baseTypeDependency);
                new DataFlowEdge(dotDeref, read);
                new ReadMemoryElementEdge(dotDeref->memoryElement, read);

                new ControlFlowEdge(read);

                return read;
            }
        }

        return readExpression(binaryOp->get_lhs_operand());

    } else if (auto commaExpr = isSgCommaOpExp(expr)) {
        readExpression(commaExpr->get_lhs_operand());
        return readExpression(commaExpr->get_rhs_operand());
    } else if (auto minusExpr = isSgMinusOp(expr)) {
        Node *input = readExpression(minusExpr->get_operand());
        if (input->getVariant() == NodeVariant::CONSTANT) {
            ConstantNode *constantInput = dynamic_cast<ConstantNode *>(input);
            constantInput->value = constantInput->value * -1;
            return constantInput;
        }
        if (input->getType().dataType == DataType::FLOAT) {
            Node *node = new FNegNode();
            new DataFlowEdge(input, node);
            new ControlFlowEdge(node);
            return node;
        } else {
            return input;
        }

    } else if (auto sizeOfExpr = isSgSizeOfOp(expr)) {
        SgExpression *input = sizeOfExpr->get_operand_expr();

        bool found = false;
        int value = 8;
        if (input){
            while (!found) {
                if (auto varRef = isSgVarRefExp(input)) {
                    found = true;
                } else if (auto dotExpr = isSgDotExp(input)) {
                    input = dotExpr->get_rhs_operand();
                } else if (auto arrowExpr = isSgArrowExp(input)) {
                    input = arrowExpr->get_rhs_operand();
                } else {
                    throw std::runtime_error("Unsupport expr in size of expr: " + input->unparseToString());
                }
            }
            SgType *inputType = input->get_type();
            assert(inputType);

            if (inputType->variantT() == V_SgArrayType) {
                SgArrayType *arrayType = isSgArrayType(inputType);
                assert(arrayType);
                int numElements = arrayType->get_number_of_elements();
                int byteWidth;
                SgType *baseType = arrayType->get_base_type()->findBaseType();
                if (baseType->variantT() == V_SgTypeUnsignedChar) {
                    byteWidth = 1;
                } else {
                    throw std::runtime_error("Unsupported base type for sizeof expression: " + baseType->unparseToString());
                }
                value = byteWidth * numElements;
            } else {
                throw std::runtime_error("Unsupported top level type for sizeof expression: " +
                                        inputType->unparseToString());
            }
        }

        TypeStruct sizeDesc = TypeStruct(DataType::INTEGER, 64);
        Node *node = new FakeConstantNode(std::to_string(value), sizeDesc);
        return node;
    } else if (auto nullExpr = isSgNullExpression(expr)) {
        return nullptr;
    } else if (auto addressOfExpr = isSgAddressOfOp(expr)) {
        SgExpression *input = addressOfExpr->get_operand();
        if (auto varRef = isSgVarRefExp(input)) {
            auto varDec = varRef->get_symbol()->get_declaration();
            return variableMapper->readVariable(varDec);
        } else if (auto arrayRef = isSgPntrArrRefExp(input)) {
            auto binaryOp = isSgBinaryOp(arrayRef);
            assert(binaryOp);
            Node *deref = getDerefNode(binaryOp);

            return deref;
        } else {
            throw std::runtime_error("Unsupported input for address of operation: " + input->unparseToString());
        }
    } else if (auto selectOp = isSgConditionalExp(expr)) {
        Node *condition = readExpression(selectOp->get_conditional_exp());

        Node *pred = condition;
        if (graphGenerator->checkArg(PROXY_PROGRAML)){
            if (condition->getVariant() != NodeVariant::COMPARISON) {
                // there's actually a few types of nodes here to
                // convert to i1
                // TODO: check which node to actually add
                // instead of always a comparison
                ComparisonNode *comp = new ComparisonNode();

                Node *constant = new FakeConstantNode("0", condition->getType());
                ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(condition, constant, comp);
                comp->edge = edge;

                // used for iterator bounds checking
                // should never happen but just in case we want to avoid nullptr
                comp->lhs = condition;
                comp->rhs = constant;

                new ControlFlowEdge(comp);

                pred = comp;
            }
        }

        Node *branch = new BranchNode();
        new ControlFlowEdge(branch);
        new DataFlowEdge(pred, branch);

        Node *a = readExpression(selectOp->get_true_exp());

        new ProgramlBranchEdge();

        MergeStartEdge *mergeStart = new MergeStartEdge();

        new RevertControlFlowEdge(branch);

        Node *b = readExpression(selectOp->get_false_exp());

        new ProgramlBranchEdge();

        new MergeEndEdge(mergeStart);

        SelectNode *select = new SelectNode();

        ArithmeticUnitEdge *arithmeticEdge = new ArithmeticUnitEdge(a, b, select);
        select->edge = arithmeticEdge;

        new ControlFlowEdge(select);
        return select;
    } else {
        throw std::runtime_error("Unsupported expression: " + expr->unparseToString());
    }
}

// aes kernel is stupidly written and has --i in conditional
// so I need this code in two places
Node *AstParser::handleIncOp(SgExpression *expr){
// cast to a unary op to get access to member variables
        SgUnaryOp *unaryOp = isSgUnaryOp(expr);
        // read from the lhs
        Node *lhs = readExpression(unaryOp->get_operand());

        // add a constant
        Node *rhs;

        Node *arithmeticNode;
        // programl (and therefore probably clang)
        // turns a -- op into a += -1
        if(graphGenerator->checkArg(PROXY_PROGRAML)){
            if (expr->variantT() == V_SgPlusPlusOp) {
                rhs = new ConstantNode(1, lhs->getType());
            } else {
                rhs = new ConstantNode(-1, lhs->getType());
            }

            // add a
            arithmeticNode = processArithmeticExpression(V_SgPlusPlusOp, lhs, rhs);
        } else {
            rhs = new ConstantNode(1, lhs->getType());

            // add the correct arithmetic node
            arithmeticNode = processArithmeticExpression(expr->variantT(), lhs, rhs);
        }


        new ControlFlowEdge(arithmeticNode);

        // and write to the lhs
        Node *writeNode = writeExpression(unaryOp->get_operand(), arithmeticNode);
        new ControlFlowEdge(writeNode);

        return writeNode;
}

// if statements don't need the comparison returned
void AstParser::handleConditional(SgExpression *expr, Node *&branchNodeOut) {
    Node *comparisonNode;
    handleConditional(expr, branchNodeOut, comparisonNode);
}

// add all nodes for the comparison to the graph
// and return the branch and comparison by ref
// branch for looping control flow
// and comparison for adding pragma nodes to
void AstParser::handleConditional(SgExpression *expr, Node *&branchNodeOut, Node *&comparisonNodeOut) {

    bool exception = false;
    if(graphGenerator->checkArg(PROXY_PROGRAML)){
        if (auto *cast = isSgCastExp(expr)) {
            exception = true;
            // programl needs a boolean cast

            Node *lhs = readExpression(cast->get_operand());

            Node *rhs = new FakeConstantNode("0", lhs->getType());

            ComparisonNode* comparisonNode = new ComparisonNode();

            comparisonNode->lhs = lhs;
            comparisonNode->rhs = rhs;

            ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, comparisonNode);
            comparisonNode->edge = edge;

            // comparisonNodeOut is just a Node*
            comparisonNodeOut = comparisonNode;
        }
    } 

    if(!exception){
        comparisonNodeOut = readExpression(expr);
    }

    assert(comparisonNodeOut);
    new ControlFlowEdge(comparisonNodeOut);

    branchNodeOut = new BranchNode();

    new DataFlowEdge(comparisonNodeOut, branchNodeOut);
    new ControlFlowEdge(branchNodeOut);
}

DerefNode *AstParser::getDerefNode(SgBinaryOp *arrayIndex) {
    return getDerefNode(arrayIndex, false);
}

DerefNode *AstParser::getDerefNode(SgBinaryOp *arrayIndex, bool ignoreSaved) {
    DerefNode *deref = nullptr;

    // if this array index hasn't been seen before in this bb
    if (ignoreSaved || !(deref = derefTracker->getDerefNode(arrayIndex))) {
        SgExpression *lhs = arrayIndex->get_lhs_operand();

        // TODO: document weird intersections of pointer and multi dereferences
        // just from reading this instead of having 2 deref nodes,
        // a single deref node is used with 2 inputs
        if (SgPntrArrRefExp *prevDerefExpr = isSgPntrArrRefExp(lhs)){
            if(!graphGenerator->checkArg(PROXY_PROGRAML)){
                Node *rhs = readExpression(arrayIndex->get_rhs_operand());
                DerefNode *deref = getDerefNode(prevDerefExpr, true);
                new DataFlowEdge(rhs, deref);
                if(!ignoreSaved){
                    derefTracker->saveDerefNode(arrayIndex, deref);
                }
                return deref;
            }
        } 


        // add a node for the address calculation
        deref = new DerefNode();
        derefTracker->saveDerefNode(arrayIndex, deref);



        bool resolvedParent = false;
        // TODO: document weird intersections of pointer and arrays
        if (lhs->variantT() == V_SgDotExp || lhs->variantT() == V_SgArrowExp) {
            SgBinaryOp *binaryOp = isSgBinaryOp(lhs);
            assert(binaryOp);
            // something happens here to do with a pointer to a pointer
            // which requires an extra load
            if(graphGenerator->checkArg(PROXY_PROGRAML)){
                DerefNode *dotNode = getDotNode(binaryOp);
                deref->memoryElement = dotNode->memoryElement;

                deref->baseTypeDependency = dotNode->baseTypeDependency;
                deref->typeDependency = dotNode;

                new ParameterLoadDataFlowEdge(dotNode, deref);
                resolvedParent = true;
            } else{
                // how we handle it:
                // ignore the dot or the arrow and just give me what was on the other side of it
                lhs = binaryOp->get_lhs_operand();
            }
        }

        if(!resolvedParent){
            // if the lhs of an array deref is an array deref
            // we can only get here if proxying programl
            if (SgPntrArrRefExp *prevDerefExpr = isSgPntrArrRefExp(lhs)) {
                DerefNode *prevDeref = getDerefNode(prevDerefExpr);

                // these make sure the types are all correct
                deref->memoryElement = prevDeref->memoryElement;
                deref->typeDependency = prevDeref;
                deref->baseTypeDependency = prevDeref->baseTypeDependency;


                // the prevDeref gives a pointer so LLVM would add an extra load
                new ParameterLoadDataFlowEdge(prevDeref, deref);
            
            // normal array dereference- lhs is the array itself
            } else if (SgVarRefExp *varRef = isSgVarRefExp(lhs)) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

                // only connect the array address node to the variable if
                // we are treating the variable as an address
                // if the variable is a memory element, it has nothing to do with
                // the address
                Node *array = variableMapper->readVariable(varDec);
                new ParameterLoadDataFlowEdge(array, deref);

                deref->memoryElement = array;
                deref->typeDependency = array;
                deref->baseTypeDependency = array;
            }  else {
                throw std::runtime_error("array indexing indexed something unknown");
            }
        }
        // get the index value
        Node *rhs = readExpression(arrayIndex->get_rhs_operand());

        new SextDataFlowEdge(rhs, deref);
        new ControlFlowEdge(deref);
    }

    return deref;
}

DerefNode *AstParser::getDotNode(SgBinaryOp *binaryOp) {

    SgExpression *rhsOp = binaryOp->get_rhs_operand();
    SgVarRefExp *rightVarRef = isSgVarRefExp(rhsOp);

    assert(rightVarRef);

    SgInitializedName *rightVar = rightVarRef->get_symbol()->get_declaration();

    SgType *structType = binaryOp->get_lhs_operand()->get_type()->findBaseType();
    StructFieldNode *structField = variableMapper->getStructField(structType, rightVar);
    
    StructAccessNode *accessNode = new StructAccessNode();
    Node *structAddressSource = nullptr;
    Node *memoryElement = nullptr;
    if (auto lhsVarRef = isSgVarRefExp(binaryOp->get_lhs_operand())) {
        SgInitializedName *leftVar = lhsVarRef->get_symbol()->get_declaration();
        structAddressSource = variableMapper->readVariable(leftVar);
        memoryElement = structAddressSource;
        new ParameterLoadDataFlowEdge(structAddressSource, accessNode);
    } else if (auto lhsArrayDeref = isSgPntrArrRefExp(binaryOp->get_lhs_operand())) {
        structAddressSource = getDerefNode(lhsArrayDeref);
        SgVarRefExp *varRef = isSgVarRefExp(lhsArrayDeref->get_lhs_operand());
        assert(varRef);
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
        assert(varDec);
        memoryElement = Edges::graphGenerator->variableMapper->readVariable(varDec);
        new DataFlowEdge(structAddressSource, accessNode);
    }
    new ControlFlowEdge(accessNode);

    StructAccessEdge *dotEdge = new StructAccessEdge(accessNode);

    accessNode->memoryElement = memoryElement;
    accessNode->setType(structField->getImmediateType());
    accessNode->baseTypeDependency = structField;

    return accessNode;
}

Node *evaluateConstantArithmetic(SgExpression *expr, Node *lhs, Node *rhs) {
    std::string stringOp = Utils::getArithmeticNodeEncoding(expr->variantT());
    ConstantNode *lhsConstant = dynamic_cast<ConstantNode *>(lhs);
    ConstantNode *rhsConstant = dynamic_cast<ConstantNode *>(rhs);

    lhsConstant->folded = true;
    rhsConstant->folded = true;

    double a = lhsConstant->value;
    double b = rhsConstant->value;
    double result;
    if (stringOp == "Multiplication") {
        result = a * b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "*" << b << "=" << result << std::endl;
    } else if (stringOp == "Addition") {
        result = a + b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "+" << b << "=" << result << std::endl;
    } else if (stringOp == "Subtraction") {
        result = a - b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "-" << b << "=" << result << std::endl;
    } else if (stringOp == "Division") {
        result = a / b;
        std::cerr << "folding:" << std::endl;
        std::cerr << a << "/" << b << "=" << result << std::endl;
    } else if (stringOp == "LeftShift") {
        result = int(a) << int(b);
        std::cerr << "folding:" << std::endl;
        std::cerr << int(a) << "<<" << int(b) << "=" << result << std::endl;
    } else if (stringOp == "RightShift") {
        result = int(a) >> int(b);
        std::cerr << "folding:" << std::endl;
        std::cerr << int(a) << ">>" << int(b) << "=" << result << std::endl;
    } else {
        throw std::runtime_error("Found unexpected arithmetic encoding: " + stringOp);
    }

    Node *node = new ConstantNode(result, lhsConstant->getType());
    return node;
}

Node *AstParser::readArithmeticExpression(SgExpression *expr) {
    if (SgBinaryOp *binaryOp = isSgBinaryOp(expr)) {
        Node *node;

        Node *lhs = readExpression(binaryOp->get_lhs_operand());
        Node *rhs = readExpression(binaryOp->get_rhs_operand());

        if (lhs->getVariant() == NodeVariant::CONSTANT && rhs->getVariant() == NodeVariant::CONSTANT) {
            ConstantNode *lhsConst = dynamic_cast<ConstantNode *>(lhs);
            ConstantNode *rhsConst = dynamic_cast<ConstantNode *>(rhs);

            if(lhsConst->canFold && rhsConst->canFold){
                return evaluateConstantArithmetic(expr, lhs, rhs);
            }
        }

        // this error catching is for unpredicted arithmetic op types
        // since my type system is string based
        // should change it eventually
        try {
            node = processArithmeticExpression(binaryOp->variantT(), lhs, rhs);
            new ControlFlowEdge(node);
        } catch (std::runtime_error e) {
            std::cout << e.what() + binaryOp->unparseToString() << std::endl;
        }
        return node;
    } else {
        throw std::runtime_error("Arithmetic node was not a binary operation: " + expr->unparseToString());
    }
}

Node *AstParser::processArithmeticExpression(VariantT variant, Node *lhs, Node *rhs) {
    ArithmeticNode *node = new ArithmeticNode(variant);

    ArithmeticUnitEdge *edge = new ArithmeticUnitEdge(lhs, rhs, node);
    node->arithmeticEdge = edge;

    return node;
}

Node *AstParser::addWrite(Node *variable, Node *rhs, Node *typeDependency) {
    assert(typeDependency);
    WriteNode *writeNode = new WriteNode(typeDependency);

    // data often changes type before it goes to a write node
    // but its a different logic than for an arithmetic unit
    // so the ImplicitCastDataFlowEdge makes sure the data type 
    // from the input data is correct
    Edge *dataflowEdge = new ImplicitCastDataFlowEdge(rhs, writeNode);

    // I don't use edge order, but we need it to replicate programl
    // the data input to a write node is edge 1
    // since when proxying programl the dataflow edge of the pointer 
    // will be edge 0
    dataflowEdge->order = 1;

    // the immediate input is used for iterator bitwidth reduction
    // rather than assuming the iterator starts at 0
    writeNode->immediateInput = rhs;

    // and make the write an input to that node
    // this reverses if proxying programl
    new WriteMemoryElementEdge(writeNode, variable);
    return writeNode;
}

Node *AstParser::writeExpression(SgNode *lhs, Node *rhs) {
    // are we writing to a variable
    if (SgVarRefExp *varRef = isSgVarRefExp(lhs)) {
        // get the declaration
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();

        // get the node that writes to this variable should connect to
        Node *variable = variableMapper->writeVariable(varDec);

        return addWrite(variable, rhs, variable);
        // if its an array index
    } else if (SgPntrArrRefExp *arrayIndex = isSgPntrArrRefExp(lhs)) {
        DerefNode *derefNode = getDerefNode(arrayIndex);
        Node *write = addWrite(derefNode->memoryElement, rhs, derefNode->baseTypeDependency);

        new DataFlowEdge(derefNode, write);

        return write;
    } else if (SgPointerDerefExp *pointerDeref = isSgPointerDerefExp(lhs)) {
        auto varRef = isSgVarRefExp(pointerDeref->get_operand());
        assert(varRef);
        auto varDec = varRef->get_symbol()->get_declaration();
        assert(varDec);

        Node *variable = variableMapper->writeVariable(varDec);

        // make a fake constant node and mark it as folded so it doesn't print
        TypeStruct pointerTypeDesc = TypeStruct(DataType::INTEGER, 64);
        ConstantNode *fakeConstant = new ConstantNode(0, pointerTypeDesc);
        fakeConstant->folded = true;

        // use the fake constant to create a read node which produces
        // a 64-bit int 
        Node *read = new ReadNode(fakeConstant);
        new ControlFlowEdge(read);

        new MemoryAddressEdge(variable, read);

        Node *write = addWrite(read, rhs, variable);
        new MemoryAddressEdge(read, write);
        return write;
    } else if (lhs->variantT() == V_SgDotExp || lhs->variantT() == V_SgArrowExp) {
        SgBinaryOp *binaryOp = isSgBinaryOp(lhs);
        assert(binaryOp);

        if(graphGenerator->checkArg(PROXY_PROGRAML)){

            DerefNode *dotNode = getDotNode(binaryOp);

            Node *write = addWrite(dotNode->memoryElement, rhs, dotNode->typeDependency);

            new DataFlowEdge(dotNode, write);
            return write;
        }
        return writeExpression(binaryOp->get_lhs_operand(), rhs);
    } else if (SgInitializedName *varDec = isSgInitializedName(lhs)) {
        Node *variable = variableMapper->writeVariable(varDec);
        return addWrite(variable, rhs, variable);
    } else {
        throw std::runtime_error("Unsupported expression to write to: " + lhs->unparseToString());
    }
}

ReturnEdge *AstParser::handleFuncDec(SgFunctionDeclaration *funcDec, Node *external) {
    FunctionStartEdge *startEdge = new FunctionStartEdge(external);
    startEdge->functionName = funcDec->get_name();

    functionStartEdgeMap[funcDec] = startEdge;

    // add possible return edge
    ReturnEdge *functionReturnEdge;

    

    if (funcDec->get_definition()) {
        SgBasicBlock *bb = funcDec->get_definition()->get_body();

        pragmaParser->parsePragmas(bb);


        graphGenerator->newBB();

        for (SgInitializedName *arg : funcDec->get_parameterList()->get_args()) {
            Node *parameter = variableMapper->createParameterVariableNode(arg);
            new ParameterInitializeEdge(parameter);
        }



        handleBB(bb->getStatementList());

        // add possible return edge
        functionReturnEdge = new ReturnEdge(functionReturn, funcDec);
    } else {
        // add possible return edge
        functionReturnEdge = new UndefinedFunctionEdge(funcDec->get_name(), graphGenerator->getFunctionID());
    }

    new ControlFlowEdge(external);

    return functionReturnEdge;
}

// inform a variable that is is initialized in the start of a for loop
// so it can consider itself for iterator bitwidth reduction
LocalScalarNode *AstParser::getIteratorFromForInit(SgStatementPtrList statements) {
    if (statements.size() > 1) {
        throw std::runtime_error("More than 1 statement in for statement init");
    }

    SgStatement *statement = statements[0];

    if (SgVariableDeclaration *varDecStatement = isSgVariableDeclaration(statement)) {
        SgInitializedNamePtrList declarations = varDecStatement->get_variables();
        if (declarations.size() > 1) {
            throw std::runtime_error("More than 1 variable declared in for statement init");
        }

        SgInitializedName *varDec = declarations[0];
        makeVariable(varDec);
        Node *variableNode = variableMapper->readVariable(varDec);

        if (LocalScalarNode *localScalar = dynamic_cast<LocalScalarNode *>(variableNode)) {
            localScalar->hasIteratorInit = true;
            return localScalar;
        }

    } else if (SgExprStatement *exprStatement = isSgExprStatement(statement)) {
        SgExpression *expr = exprStatement->get_expression();
        if (SgCommaOpExp *commaExpr = isSgCommaOpExp(expr)) {
            // if comma op, check the left side of the comma for the iterator
            // why would you use a comma in a for init I don't know
            // but they do is aes256, and the iterator is on the left
            expr = commaExpr->get_lhs_operand();
        }

        if (SgAssignOp *assignOp = isSgAssignOp(expr)) {
            if (SgVarRefExp *varRef = isSgVarRefExp(assignOp->get_lhs_operand())) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
                Node *variableNode = variableMapper->readVariable(varDec);

                if (LocalScalarNode *localScalar = dynamic_cast<LocalScalarNode *>(variableNode)) {
                    localScalar->hasIteratorInit = true;
                    return localScalar;
                }

                if (ParameterScalarNode *paramScalar = dynamic_cast<ParameterScalarNode *>(variableNode)) {
                    return nullptr;
                }
            }
        }
    }

    throw std::runtime_error("Unsupported for init statement found: " + statement->unparseToString());
}

FunctionStartEdge *AstParser::getFunctionStartEdge(SgFunctionDeclaration *funcDec) {
    if (functionStartEdgeMap.count(funcDec)) {
        return functionStartEdgeMap[funcDec];
    }

    throw std::runtime_error("Tried to get the start edge of function, but it wasn't in the map: " +
                             funcDec->get_name());
}
ReturnEdge *AstParser::getFunctionReturnEdge(SgFunctionDeclaration *funcDec) {
    if (functionReturnEdgeMap.count(funcDec)) {
        return functionReturnEdgeMap[funcDec];
    }

    throw std::runtime_error("Tried to get the return edge of function, but it wasn't in the map: " +
                             funcDec->get_name());
}

void AstParser::parseAst(SgFunctionDefinition *topLevelFuncDef) {
    SgFunctionDeclaration *topLevelFuncDec = topLevelFuncDef->get_declaration();
    graphGenerator->setGroupName("External");
    graphGenerator->setFuncDec(topLevelFuncDec);

    //inline on this dataset basically means do the unroll pragmas apply
    graphGenerator->setFuncInlined(topLevelFuncDec, true);


    Node *external = new ExternalNode(graphGenerator);

    graphGenerator->newBB();

    graphGenerator->setGroupName(topLevelFuncDec->get_name());

    handleFuncDec(topLevelFuncDec, external);
    // change how array parameters are handled
    // since top level array params are different
    variableMapper->finishedMain = true;

    while (!functionDecsNeeded.empty()) {
        SgFunctionDeclaration *funcDec = functionDecsNeeded.front();
        functionDecsNeeded.pop();

        //set default, this will be overridden later if true
        graphGenerator->setFuncInlined(funcDec, false);


        graphGenerator->setFuncDec(funcDec);

        if (functionDecsComplete.count(funcDec)) {
            continue;
        }
        functionDecsComplete.insert(funcDec);

        graphGenerator->setGroupName(funcDec->get_name());
        graphGenerator->enterNewFunction();



        ReturnEdge *functionReturnEdge = handleFuncDec(funcDec, external);



        functionReturnEdgeMap[funcDec] = functionReturnEdge;
    }

    pragmaParser->parseInlinePragmas(functionDecsComplete);

    while (!pragmaParser->inlinedFunctions.empty()) {
        SgFunctionDeclaration *funcDec = pragmaParser->inlinedFunctions.front();
        pragmaParser->inlinedFunctions.pop();

        graphGenerator->setFuncInlined(funcDec, true);

        Node *pragma = nullptr;

        //if the function was inlined on the graph, no call nodes
        if(decsToCalls.count(funcDec)){
            for (FunctionCallNode *funcCall : decsToCalls[funcDec]) {
                graphGenerator->stateNode = funcCall;
                if(!pragma){
                    pragma = new InlinedFunctionPragmaNode();
                }
                new InlineFunctionPragmaEdge(pragma, funcCall);
            }
        }
    }


    graphGenerator->registerCalls(topLevelFuncDec, 1);
    for(SgFunctionDeclaration *funcDec : functionDecsComplete){
        for (FunctionCallNode *funcCall : decsToCalls[funcDec]) {
            graphGenerator->registerCalls(funcDec, funcCall->unrollFactor.full);
        }
    }    

}

} // namespace Balor
