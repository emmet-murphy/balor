#include "edge.h"
#include "args.h"

namespace Balor {

GraphGenerator *Edges::graphGenerator = nullptr;
Node *Edges::previousControlFlowNode = nullptr;
std::queue<Edge *> Edges::previousControlFlowNodeChangeListeners;

Node *Edges::getPreviousControlFlowNode() { return previousControlFlowNode; }

// when you add a node to the control flow
// sometimes other nodes care about what the next node is
// so you have to update the listeners after changing the control flow
// so they can react appropriately
void Edges::updatePreviousControlFlowNode(Node *node) {
    previousControlFlowNode = node;
    while (!previousControlFlowNodeChangeListeners.empty()) {
        Edge *listener = previousControlFlowNodeChangeListeners.front();
        previousControlFlowNodeChangeListeners.pop();
        listener->runDeferred();
    }
}

void Edges::addPreviousControlFlowNodeChangeListener(Edge *edge) { previousControlFlowNodeChangeListeners.push(edge); }

Edge::Edge(Node *source, Node *destination) : source(source), destination(destination) {
    Edges::graphGenerator->edges.push_back(this);

    // give to unique pointer to manage memory automatically
    std::unique_ptr<Edge> edge_unique = std::unique_ptr<Edge>(this);
    // pass to vector on object so it passes out of scope at the right time
    Edges::graphGenerator->edges_unq.push_back(std::move(edge_unique));
}

WriteMemoryElementEdge::WriteMemoryElementEdge(Node *source, Node *destination) : Edge(source, destination) {
    // if the destination of a write is a local scalar
    if (destination->getVariant() == NodeVariant::LOCAL_SCALAR) {
        if (WriteNode *write = dynamic_cast<WriteNode *>(source)) {
            // get the input to the write
            Node *immediateInput = write->immediateInput;

            LocalScalarNode *scalar = dynamic_cast<LocalScalarNode *>(destination);
            // and tell the scalar something is writing to it
            // to enable iterator bitwidth reduction
            scalar->addBound(immediateInput);
        }
    }
}

void ControlFlowEdge::run() {
    // I was interested to see if we needed control flow edges between every node or only
    // actual control flow affecting things
    // but it reduced the accuracy a little bit
    // it was at an early stage though so maybe worth investigating again
    if (Edges::graphGenerator->checkArg(ONLY_MEMORY_CONTROL_FLOW)) {
        bool memoryNode = destination->getVariant() == NodeVariant::MEMORY;
        bool branchNode = destination->getVariant() == NodeVariant::BRANCH;
        bool externalNode = destination->getVariant() == NodeVariant::EXTERNAL;
        bool returnNode = destination->getVariant() == NodeVariant::RETURN;
        bool callNode = destination->getVariant() == NodeVariant::CALL;

        if (!(memoryNode || branchNode || externalNode || returnNode || callNode)) {
            return;
        }
    }

    // sub control flow edge is an actual control flow edge on the graph
    Edges::printSubControlFlowEdge(Edges::getPreviousControlFlowNode(), destination, false);
    Edges::updatePreviousControlFlowNode(destination);
}

void UnrollPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 0);
    }
}

void ArrayPartitionPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 1);
    }
}

void ResourceAllocationPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 2);
    }
}

void InlineFunctionPragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 3);
    }
}

void PipelinePragmaEdge::run() {
    if (!Edges::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        Edges::printPragmaEdge(source, destination, 4);
    }
}

void DataFlowEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    bool sourceIsConstant = source->getVariant() == NodeVariant::CONSTANT;
    bool sourceIsParameter = source->getVariant() == NodeVariant::ALLOCA_INITIALIZER;
    bool sourceIsGlobalArray = source->getVariant() == NodeVariant::GLOBAL_ARRAY;
    bool globalArrayIsValue = Edges::graphGenerator->checkArg(PROXY_PROGRAML);
    bool sourceIsValue = sourceIsConstant || sourceIsParameter || (sourceIsGlobalArray && globalArrayIsValue);
    bool hideValues = Edges::graphGenerator->checkArg(HIDE_VALUES);
    bool absorbTypes = Edges::graphGenerator->checkArg(ABSORB_TYPES);

    bool hasSource = !(sourceIsValue && hideValues);

    bool sourceToDest = absorbTypes && hasSource;
    bool sourceToType = (!absorbTypes) && hasSource;
    bool typeToDest = !absorbTypes;

    // dataflow edges are added between lots of types of nodes
    // and have to add type nodes if proxying programl
    // so the first question is:
    // should I add a type node and connect it to the destination?
    if (typeToDest) {
        TypeStruct sourceType = source->getImmediateType();

        Node *typeNode = new TypeNode(sourceType, sourceIsConstant);
        typeNode->print();


        Edges::printSubDataFlowEdge(typeNode, destination, order);
        if (sourceToType) {
            Edges::printSubDataFlowEdge(source, typeNode);
        }
    } else if (sourceToDest) {
        Edges::printSubDataFlowEdge(source, destination, order);
    }
}

void ReadMemoryElementEdge::run() {
    bool allocas = !Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS);
    bool elementIsLocalScalar = source->getVariant() == NodeVariant::LOCAL_SCALAR;
    bool elementIsParameterScalar = source->getVariant() == NodeVariant::PARAMETER_SCALAR;
    bool elementIsScalar = elementIsLocalScalar || elementIsParameterScalar;

    // array elements have memory address edges from the getelemptr node
    // scalar elements have them directly from the memory element
    // which is presenting itself as an alloca
    if (allocas) {
        if (elementIsScalar) {
            (new MemoryAddressEdge(source, destination))->run();
        }
    } else {
        MemoryAddressEdge *edge = new MemoryAddressEdge(source, destination);
        edge->typeDependency = source;
        edge->run();
    }
}

void WriteMemoryElementEdge::run() {
    bool allocas = !Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS);
    bool elementIsLocalScalar = destination->getVariant() == NodeVariant::LOCAL_SCALAR;
    bool elementIsParameterScalar = destination->getVariant() == NodeVariant::PARAMETER_SCALAR;
    bool elementIsScalar = elementIsLocalScalar || elementIsParameterScalar;

    // array elements have memory address edges from the getelemptr node
    // scalar elements have them directly from the memory element
    // which is presenting itself as an alloca
    //
    // also for writes, the edge still still comes from the alloca
    // so its reversed for programl
    if (allocas) {
        if (elementIsScalar) {
            (new MemoryAddressEdge(destination, source))->run();
        }
    } else {
        MemoryAddressEdge *edge = new MemoryAddressEdge(source, destination);
        edge->typeDependency = destination;
        edge->run();
    }
}

void MemoryAddressEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    bool sourceIsConstant = source->getVariant() == NodeVariant::CONSTANT;
    bool sourceIsParameter = source->getVariant() == NodeVariant::ALLOCA_INITIALIZER;
    bool sourceIsGlobalArray = source->getVariant() == NodeVariant::GLOBAL_ARRAY;
    bool globalArrayIsValue = Edges::graphGenerator->checkArg(PROXY_PROGRAML);
    bool sourceIsValue = sourceIsConstant || sourceIsParameter || (sourceIsGlobalArray && globalArrayIsValue);
    bool hideValues = Edges::graphGenerator->checkArg(HIDE_VALUES);
    bool absorbTypes = Edges::graphGenerator->checkArg(ABSORB_TYPES);

    bool hasSource = !(sourceIsValue && hideValues);

    bool sourceToDest = absorbTypes && hasSource;
    bool sourceToType = (!absorbTypes) && hasSource;
    bool typeToDest = !absorbTypes;

    if (typeToDest) {
        TypeStruct sourceType = source->getImmediateType();

        Node *typeNode;
        if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            typeNode = new TypeNode(source->getImmediateType(), sourceIsConstant);
            typeNode->print();
        } else {
            typeNode = new TypeNode(getElemType(), sourceIsConstant);
            typeNode->print();
        }

        Edges::printSubMemoryAddressEdge(typeNode, destination);
        if (sourceToType) {
            Edges::printSubMemoryAddressEdge(source, typeNode);
        }
    } else if (sourceToDest) {
        Edges::printSubMemoryAddressEdge(source, destination);
    }
}

void SpecifyAddressEdge::run() {
    Edges::graphGenerator->stateNode = destination;
    if (!Edges::graphGenerator->checkArg(ABSORB_TYPES)) {
        Node *typeNode = new TypeNode(source->getType(), false);
        typeNode->print();

        Edges::printSubMemoryAddressEdge(source, typeNode);
        Edges::printSubMemoryAddressEdge(typeNode, destination);
    } else {
        Edges::printSubMemoryAddressEdge(source, destination);
    }
}

TypeStruct MemoryAddressEdge::getElemType() {
    if (typeDependency) {
        return typeDependency->getType();
    }
    throw std::runtime_error("Memory element edge didn't have its type dependency set");
}

void ProgramlBranchEdge::run() {
    if (!Edges::graphGenerator->checkArg(REMOVE_SINGLE_TARGET_BRANCHES)) {
        Node *branch = new BranchNode();
        branch->print();

        (new ControlFlowEdge(branch))->run();
    }
}

void SextDataFlowEdge::run() {
    bool addSexts = true;
    int bitGoal = 64;

    // if we can just ignore sexts, ignore them
    if (Edges::graphGenerator->checkArg(REMOVE_SEXTS)) {
        addSexts = false;
    } else {

        // programl treats pointers as 64 bit
        // but I decided to treat them as 32 bit for hardware
        // not sure how true that actually is
        // do BRAMs normally take 32 bit addresses?
        if (Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
            bitGoal = 32;
        }

        // this is set if a 32 bit number needs to be sign extended
        // to 64 before adding to a 64 bit number
        if (widthOverridden) {
            bitGoal = newWidth;
        }

        // don't add a sext before a constant, just change the bitwidth
        // of the constant
        if (source->getVariant() == NodeVariant::CONSTANT) {
            source->setType(TypeStruct(source->getSextType().dataType, bitGoal));
            addSexts = false;
        } else {
            // only add the sext if the incoming data is below what it needs to be
            TypeStruct type = source->getSextType();
            if (type.bitwidth >= bitGoal) {
                addSexts = false;
            }
        }
    }

    // if in the incoming data is below the needed goal
    // add a sext
    if (addSexts) {
        Edges::graphGenerator->stateNode = destination;
        Node *sext = new SextNode(bitGoal, source->getSextType().isUnsigned);
        sext->print();

        (new DataFlowEdge(source, sext))->run();
        Edge *edge = new DataFlowEdge(sext, destination);
        edge->order = 1;
        edge->run();

        (new ControlFlowEdge(sext))->run();
    } else {
        // otherwise just print
        Edge *edge = new DataFlowEdge(source, destination);
        edge->order = 1;
        edge->run();
    }
}

void ParameterLoadDataFlowEdge::run() {
    // parameter loads only added for allocas
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        Edges::graphGenerator->stateNode = destination;
        bool externalArray = source->getVariant() == NodeVariant::EXTERNAL_ARRAY;
        bool parameterArray = source->getVariant() == NodeVariant::PARAMETER_ARRAY;
        if (externalArray || parameterArray) {

            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
            Node *read = new ReadNode(pointerType);
            read->print();

            (new ControlFlowEdge(read))->run();
            (new MemoryAddressEdge(source, read))->run();
            (new DataFlowEdge(read, destination))->run();
        } else {
            (new MemoryAddressEdge(source, destination))->run();

            if (Nodes::graphGenerator->checkArg(PROXY_PROGRAML)) {
                TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                Node *node = new FakeConstantNode("Local Array Stack Pointer", pointerType);
                node->print();

                (new DataFlowEdge(node, destination))->run();
            }
        }
    }
}

void ParameterInitializeEdge::run() {
    // parameters to a top level function
    // have a lot of processing in programl
    // this just replicates that
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        Edges::graphGenerator->stateNode = source;

        // add a control flow edge to the alloca
        (new ControlFlowEdge(source))->run();

        Node *previousNode = source;
        // if the alloca is a struct
        if (source->getVariant() == NodeVariant::STRUCT) {
            // add a bitcast
            Node *bitcast = new BitcastNode();
            bitcast->print();
            (new ControlFlowEdge(bitcast))->run();

            (new DataFlowEdge(source, bitcast))->run();

            previousNode = bitcast;
        }

        // add a write node of type 64 bit int
        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
        Node *store = new WriteNode(pointerType);

        store->print();

        // add a dataflow edge of the incoming pointer data and a control flow edge
        // to store the incoming parameter
        (new ControlFlowEdge(store))->run();
        (new MemoryAddressEdge(previousNode, store))->run();

        Node *initialValueNode;
        // if it was a scalar, the initializer has the type of the scalar
        if (source->getVariant() == NodeVariant::PARAMETER_SCALAR) {
            initialValueNode = new AllocaInitializerNode(source->getType());
        } else {
        // if array the initializer is a pointer so 64-bit int
            TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
            initialValueNode = new AllocaInitializerNode(pointerType);
        }

        // print the initalizer
        initialValueNode->print();

        // and add a dataflow edge from it into the write
        (new DataFlowEdge(initialValueNode, store, 1))->run();

        Edges::updatePreviousControlFlowNode(store);
    }
}


// used when declaring scalar local variables
// as programl adds them to the control flow
void VariableDeclareEdge::run() {
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        (new ControlFlowEdge(source))->run();
    }
}

void ReturnEdge::run() {
    Node *returnNode = getNode();
    Edges::graphGenerator->newBB();
    Edges::graphGenerator->stateNode = nullptr;

    returnNode->bbID = Edges::graphGenerator->getBBID();

    // this function sets the state variable different depending on
    // if the function we are returning from is defined or not
    setNodeVariables(returnNode);
    returnNode->print();

    Edges::graphGenerator->newBB();

    (new ControlFlowEdge(returnNode))->run();

    // add a return edge to each call location
    for (Node *returnLocation : returnLocations) {
        Edges::printSubFunctionCallEdge(returnNode, returnLocation);
    }

    // does the function return a value? if so where that value is calculated 
    // should connect to the return node
    // might need a cast e.g. the calculated value is an int but the function returns a float
    if (functionReturn) {
        SgType *returnType = funcDec->get_orig_return_type()->findBaseType();
        TypeStruct returnTypeDesc = TypeStruct(returnType->unparseToString());
        returnNode->setType(returnTypeDesc);
        (new ImplicitCastDataFlowEdge(functionReturn, returnNode))->run();
    } else {
        TypeStruct returnTypeDesc = TypeStruct();
        returnNode->setType(returnTypeDesc);
    }   
}


// set the state variables based on the last node processed
void ReturnEdge::setNodeVariables(Node *node) {
    Node *pred = Edges::getPreviousControlFlowNode();
    node->functionID = pred->functionID;
    node->groupName = pred->groupName;
}

// a "returnEdge" is actually all return edges to all call locations
// but the node the edge comes from should only be added once
Node *ReturnEdge::getNode() {
    if (!privateNode) {
        privateNode = new ReturnNode();
    }
    return privateNode;
}

// if undefined, these variables will be stored in the edge itself
void UndefinedFunctionEdge::setNodeVariables(Node *node) {
    node->functionID = funcID;
    node->groupName = functionName;
}

Node *UndefinedFunctionEdge::getNode() {
    if (!privateNode) {
        privateNode = new UndefinedFunctionNode(functionName);
    }
    return privateNode;
}

void LoopBackEdge::run() {
    (new BackControlFlowEdge(preLoopEdge->loopConditionStart))->run();
    Edges::updatePreviousControlFlowNode(branch);
}

void MergeStartEdge::run(){
    source1 = Edges::getPreviousControlFlowNode(); 
}

// Run doesn't know where to merge to yet
// So it stores the second place to merge from in source2
// and tells the Edges class to call runDeferred after
// a control flow edge is added from source2
void MergeEndEdge::run() {
    source2 = Edges::getPreviousControlFlowNode();
    Edges::addPreviousControlFlowNodeChangeListener(this);
}

// After a control flow edge is added from source2
// we can get the current control flow node
// and add an edge from source1 to it
void MergeEndEdge::runDeferred() {
    Node *destination = Edges::getPreviousControlFlowNode();
    Node *source1 = openEdge->source1;

    // only add an extra control flow edge if something happened
    // between starting and ending the merge
    if (source1 != source2) {
        Edges::printSubControlFlowEdge(source1, destination);
    }
}

void FunctionStartEdge::run() {
    // all functions should have the external node as their predecessor
    Edges::updatePreviousControlFlowNode(external);
    Edges::addPreviousControlFlowNodeChangeListener(this);
}

void FunctionStartEdge::runDeferred() {
    Node *startNode = Edges::getPreviousControlFlowNode();
    // start at 1 as there's already an edge from external
    int edgeID = 1;
    for (Node *callLocation : callLocations) {
        Edges::printSubFunctionCallEdge(callLocation, startNode, edgeID);
        edgeID++;
    }
}

// do either programl function preprocessing
// or connect mem elements in simple way
void FunctionCallEdge::run() {
    if (!Edges::graphGenerator->checkArg(INLINE_FUNCTIONS)) {

        std::cerr << "function call edge" << std::endl;

        Edges::graphGenerator->stateNode = funcCallNode;

        int parameterEdgeID = 0;
        // for each parameter
        for (SgExpression *expr : parameters) {
            std::cerr << expr->unparseToString() << std::endl;
            // get the original param type
            SgInitializedName *originalParam = funcDec->get_parameterList()->get_args()[parameterEdgeID];
            SgType *paramType = originalParam->get_type();

            // and build a type struct
            TypeStruct parameterTypeDesc = TypeStruct(paramType->findBaseType()->unparseToString());
            // if its an array type, we actually don't want the real pointer type
            // we want a void pointer (because this is what programl does)
            if (paramType->variantT() == V_SgArrayType || paramType->variantT() == V_SgPointerType) {
                parameterTypeDesc = TypeStruct(DataType::INTEGER, 64);
            }

            // add a fake constant node to pull a data type node from
            ConstantNode *paramTypeDependency = new FakeConstantNode("", parameterTypeDesc);
            paramTypeDependency->folded = true;

            // reset the vector of nodes and vector of edges
            // so we can call print from here without duplicate
            // (we are currently iterating over a clone of the edge vector so resetting doesn't
            // change the current execution)
            // we need to do this since the original AST parsing doesn't parse
            // parameters, since for many cases its too complicated
            Edges::graphGenerator->nodes = std::vector<Node *>();
            Edges::graphGenerator->edges = std::vector<Edge *>();

            // exceptions are types of parameters we cannot simply call read expression
            bool foundException = false;
            // if the value passed to the function is a variable
            if (SgVarRefExp *varRef = isSgVarRefExp(expr)) {
                SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
                // get the variable node
                Node *variableRead = Edges::graphGenerator->variableMapper->readVariable(varDec);
                // if we can ignore the func processing
                if (Edges::graphGenerator->checkArg(DROP_FUNC_CALL_PROC)) {
                    NodeVariant variant = variableRead->getVariant();
                    bool localArray = variant == NodeVariant::LOCAL_ARRAY;
                    bool externalArray = variant == NodeVariant::EXTERNAL_ARRAY;
                    bool parameterScalar = variant == NodeVariant::PARAMETER_SCALAR;
                    bool localScalar = variant == NodeVariant::LOCAL_SCALAR;
                    // connect the memory element to the function call without a read node
                    if (localArray || externalArray || parameterScalar || localScalar) {
                        foundException = true;
                        (new ReadMemoryElementEdge(variableRead, funcCallNode))->run();
                        (new WriteMemoryElementEdge(funcCallNode, variableRead))->run();
                    }
                } else {
                    // if we have to do programl function preprocessing
                    bool isExternal = variableRead->getVariant() == NodeVariant::EXTERNAL_ARRAY;
                    bool isParameter = variableRead->getVariant() == NodeVariant::PARAMETER_ARRAY;

                    // array parameters are void pointers to a pointer
                    if (isExternal || isParameter) {
                        
                        foundException = true;

                        // add the read node to get the actual pointer
                        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                        Node *readNode = new ReadNode(pointerType);
                        readNode->print();

                        (new ControlFlowEdge(readNode))->run();
                        (new MemoryAddressEdge(variableRead, readNode))->run();

                        // and connec the pointer with implicit dataflow edge
                        Edge *edge = new ImplicitCastDataFlowEdge(readNode, funcCallNode, paramTypeDependency);
                        edge->order = parameterEdgeID;
                        edge->run();
                        parameterEdgeID++;

                    } else if (variableRead->getVariant() == NodeVariant::LOCAL_ARRAY) {
                        // if its a local array
                        // its not a pointer to a pointer
                        // but we do need to cast it to a void pointer
                        // using a deref node 


                        foundException = true;

                        Node *derefNode = new DerefNode();
                        derefNode->print();

                        (new ControlFlowEdge(derefNode))->run();

                        (new MemoryAddressEdge(variableRead, derefNode))->run();

                        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
                        Node *valueNode = new FakeConstantNode("0?", pointerType);
                        valueNode->print();

                        // print first unknown edge
                        (new DataFlowEdge(valueNode, derefNode))->run();

                        // print second unknown edge
                        (new DataFlowEdge(valueNode, derefNode))->run();

                        Edge *edge = new ImplicitCastDataFlowEdge(derefNode, funcCallNode, paramTypeDependency);
                        edge->order = parameterEdgeID;
                        edge->run();
                        parameterEdgeID++;
                    }
                }
            }


            //otherwise we can just parse the AST of whats in the parameters normally
            // be calling read expression
            Edges::graphGenerator->derefTracker->makeNewDerefMap();
            if (!foundException) {
                std::cerr << "not an exception" << std::endl;

                Node *parameterRead = Edges::graphGenerator->astParser->readExpression(expr);

                std::cerr << "parameter read" << std::endl;

                std::cerr << "printing nodes" << std::endl;
                //clone 
                std::vector<Node *> nodesFrozen = Edges::graphGenerator->nodes;
                for (Node *node : nodesFrozen) {
                    node->print();
                }

                std::cerr << "printing edges" << std::endl;


                //clone 
                std::vector<Edge *> edgesFrozen = Edges::graphGenerator->edges;
                for (Edge *edge : edgesFrozen) {
                    edge->run();
                }

                std::cerr << "make implicit cast dataflow edge" << std::endl;

                Edge *edge = new ImplicitCastDataFlowEdge(parameterRead, funcCallNode, paramTypeDependency);
                edge->order = parameterEdgeID;
                edge->run();
                parameterEdgeID++;
            }
        }

        std::cerr << "params complete" << std::endl;

        // now actually execute the function call
        (new ControlFlowEdge(funcCallNode))->run();

        // and record the call in start and end edges so it connects to the function body
        FunctionStartEdge *startEdge = Edges::graphGenerator->astParser->getFunctionStartEdge(funcDec);
        startEdge->callLocations.push_back(funcCallNode);

        ReturnEdge *returnEdge = Edges::graphGenerator->astParser->getFunctionReturnEdge(funcDec);
        returnEdge->returnLocations.push_back(funcCallNode);
    }
}

// we want this edge to go from destination to source for aesthetic reasons of how dot files work
// but we mark it as a back edge in both style type and print direction
// for GNN edges they're all bi-directional
void BackControlFlowEdge::run() {
    Edges::printSubControlFlowEdge(destination, Edges::getPreviousControlFlowNode(), true);
    Edges::updatePreviousControlFlowNode(destination);
}

void ArithmeticUnitEdge::run() {
    TypeStruct lhsType = lhs->getSextType();
    TypeStruct rhsType = rhs->getSextType();

    // this is to proxy programl not supporting all bitwidths at all operations
    // it returns 32 for xor or bitwise xor if proxying programl
    int unitMinBitwidth = unit->minBitwidth();

    if (lhsType.stringOverride || rhsType.stringOverride) {
        throw std::runtime_error("Overridden types cannot be used in an arithmetic unit: " + lhsType.overriddenString +
                                 " " + rhsType.overriddenString);
    }

    // in programl/clang
    // signed ICMP has a minimum of i32
    // unsigned ICMP has a minimum of i8
    // for unknown reasons???
    if(Edges::graphGenerator->checkArg(PROXY_PROGRAML)){
        if (!lhsType.isUnsigned || !rhsType.isUnsigned) {
            if (ComparisonNode *comparison = dynamic_cast<ComparisonNode *>(unit)) {
                unitMinBitwidth = 32;
            }
        }
    }


    if (lhsType.dataType == DataType::FLOAT && rhsType.dataType == DataType::FLOAT) {
        // no cast
    } else if (lhsType.dataType == DataType::INTEGER && rhsType.dataType == DataType::INTEGER) {
        // no cast
    } else if (lhsType.dataType == DataType::INTEGER && rhsType.dataType == DataType::FLOAT) {
        // cast lhs to float
        Node *node = new CastToFloatNode(rhsType);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(lhs, node))->run();
        (new DataFlowEdge(node, unit))->run();

        (new DataFlowEdge(rhs, unit, 1))->run();
        return;
    } else if (lhsType.dataType == DataType::FLOAT && rhsType.dataType == DataType::INTEGER) {
        // cast rhs to float
        Node *node = new CastToFloatNode(lhsType);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(rhs, node, 1))->run();
        (new DataFlowEdge(node, unit))->run();

        (new DataFlowEdge(lhs, unit))->run();
        return;
    }

    int maxBitwidth = lhsType.bitwidth > rhsType.bitwidth ? lhsType.bitwidth : rhsType.bitwidth;
    maxBitwidth = maxBitwidth > unitMinBitwidth ? maxBitwidth : unitMinBitwidth;

    if (lhsType.bitwidth < maxBitwidth) {
        (new SextDataFlowEdge(lhs, unit, maxBitwidth))->run();
    } else {
        (new DataFlowEdge(lhs, unit))->run();
    }

    if (rhsType.bitwidth < maxBitwidth) {
        Edge *edge = new SextDataFlowEdge(rhs, unit, maxBitwidth);
        edge->order = 1;
        edge->run();
    } else {
        (new DataFlowEdge(rhs, unit, 1))->run();
    }
}

// result of cast decisions based on input datatypes
// plus does the compiler support all ops at all bitwidths
TypeStruct ArithmeticUnitEdge::getType() {
    DataType outputType;
    TypeStruct lhsType = lhs->getSextType();
    TypeStruct rhsType = rhs->getSextType();
    int unitMinBitwidth = unit->minBitwidth();
    if (lhsType.dataType == DataType::FLOAT || rhsType.dataType == DataType::FLOAT) {
        outputType = DataType::FLOAT;
    } else {
        outputType = DataType::INTEGER;
    }

    int maxBitwidth = lhsType.bitwidth > rhsType.bitwidth ? lhsType.bitwidth : rhsType.bitwidth;
    maxBitwidth = maxBitwidth > unitMinBitwidth ? maxBitwidth : unitMinBitwidth;

    return TypeStruct(outputType, maxBitwidth);
}

void ImplicitCastDataFlowEdge::run() {
    TypeStruct type = typeDependency->getType();
    if (type.dataType == DataType::FLOAT && source->getType().dataType == DataType::INTEGER) {
        Node *node = new CastToFloatNode(type);
        node->print();
        (new ControlFlowEdge(node))->run();

        (new DataFlowEdge(source, node))->run();
        (new DataFlowEdge(node, destination, order))->run();
        return;
    }

    if(!Edges::graphGenerator->checkArg(REMOVE_SEXTS)){
        int incomingBits = source->getType().bitwidth;
        int castBits = type.bitwidth;
        if (incomingBits > castBits) {
            Node *truncate = new TruncateNode(type);
            truncate->print();
            (new DataFlowEdge(source, truncate))->run();
            (new DataFlowEdge(truncate, destination, order))->run();
            return;
        } else if (incomingBits < castBits) {
            Edge *edge = new SextDataFlowEdge(source, destination, castBits);
            edge->order = order;
            edge->run();
            return;
        } 
    } 

    (new DataFlowEdge(source, destination, order))->run();
}

void StructAccessEdge::run() {
    if(Edges::graphGenerator->checkArg(PROXY_PROGRAML)){
        Edges::graphGenerator->stateNode = accessNode;
        TypeStruct intType = TypeStruct(DataType::INTEGER, 32);
        Node *startAddress = new ConstantNode(0, intType);
        startAddress->print();
        (new DataFlowEdge(startAddress, accessNode))->run();

        Node *index = new FakeConstantNode("struct index", intType);
        index->print();
        (new DataFlowEdge(index, accessNode))->run();
    }
}


} // namespace Balor