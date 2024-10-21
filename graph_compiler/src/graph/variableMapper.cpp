

#include "variableMapper.h"

namespace Balor {

void setNumElements(Node* node, SgArrayType* arrayType) {
    int totalNumElements = 1;
    
    int nextNumElements;

    // last dim
    nextNumElements = arrayType->get_number_of_elements();
    totalNumElements *= nextNumElements;
    node->numElements0 = nextNumElements;

    SgType* baseType = arrayType->get_base_type();
    arrayType = isSgArrayType(baseType);

    // if two dims
    if(arrayType){
        nextNumElements = arrayType->get_number_of_elements();
        totalNumElements *= nextNumElements;
        node->numElements1 = nextNumElements;

        SgType* baseType = arrayType->get_base_type();
        arrayType = isSgArrayType(baseType);

        // if three dims
        if(arrayType){
            nextNumElements = arrayType->get_number_of_elements();
            totalNumElements *= nextNumElements;
            node->numElements2 = nextNumElements;

            SgType* baseType = arrayType->get_base_type();
            arrayType = isSgArrayType(baseType);

            // if four dims
            if(arrayType){
                nextNumElements = arrayType->get_number_of_elements();
                totalNumElements *= nextNumElements;
                node->numElements3 = nextNumElements;

                SgType* baseType = arrayType->get_base_type();
                arrayType = isSgArrayType(baseType);

                // if 5 dims
                if(arrayType){
                    nextNumElements = arrayType->get_number_of_elements();
                    totalNumElements *= nextNumElements;
                    node->numElements4 = nextNumElements;

                    SgType* baseType = arrayType->get_base_type();
                    arrayType = isSgArrayType(baseType);                
                }
            }
        }
    }
    node->totalNumElements = totalNumElements;
}

void VariableMapper::addArrayPragmas(const std::string &variableName, Node *pointerNode) {
    if (resourceTypeMap.count(variableName)) {
        PragmaNode *pragma;
        if (resourceTypeMap[variableName] == "RAM_2P_BRAM") {
            pragma = new Bram2P_ResourceAllocationPragmaNode();
            pointerNode->resourceType = "bram_2P";
        } else if (resourceTypeMap[variableName] == "RAM_1P_BRAM"){
            pragma = new Bram1P_ResourceAllocationPragmaNode();
            pointerNode->resourceType = "bram_1P";
        } else {
            throw std::runtime_error("Unknown resource binding: " + resourceTypeMap[variableName]);
        }

        new ResourceAllocationPragmaEdge(pragma, pointerNode);
    }

    if (arrayPartitionMap.count(variableName)) {
        // can have multiple array partition pragmas per variable
        // if they partition different dims
        while(!arrayPartitionMap[variableName].empty()){

            std::tuple<std::string, int, int> partitionData = arrayPartitionMap[variableName].front();
            arrayPartitionMap[variableName].pop();

            std::string type = std::get<0>(partitionData);
            int factor = std::get<1>(partitionData);
            int dim = std::get<2>(partitionData);

            PragmaNode *pragma = new ArrayPartitionPragmaNode(type, factor, dim);
            if (type == "complete") {
                if(dim == 1){
                    pointerNode->partitionFactor1 = 1;
                    pointerNode->partitionType1 = "complete";
                } else if(dim == 2){
                    pointerNode->partitionFactor2 = 1;
                    pointerNode->partitionType2 = "complete";
                } else {
                    pointerNode->partitionFactor3 = 1;
                    pointerNode->partitionType3 = "complete";
                }
            } else if (type == "cyclic" || type == "block") {
                if (factor > 1) {
                    if(dim == 1){
                        pointerNode->partitionFactor1 = factor;
                        pointerNode->partitionType1 = type;
                    } else if(dim == 2){
                        pointerNode->partitionFactor2 = factor;
                        pointerNode->partitionType2 = type;
                    } else {
                        pointerNode->partitionFactor3 = factor;
                        pointerNode->partitionType3 = type;                   
                    }
                }
            } else {
                throw std::runtime_error("Unrecognized partition type: " + type);
            }

            new ArrayPartitionPragmaEdge(pragma, pointerNode);
        }
    }
}

Node *VariableMapper::createLocalVariableNode(SgInitializedName *variable) {

    SgType *variableType = variable->get_type();
    if (variableType->variantT() == V_SgPointerType) {
        TypeStruct pointerType = TypeStruct(DataType::INTEGER, 64);
        LocalArrayNode *node = new LocalArrayNode(variable->get_name(), pointerType);

        variableToReadNode[variable] = node;
        variableToWriteNode[variable] = node;

        node->setType(pointerType);
        return node;
    }
    if (variableType->variantT() == V_SgArrayType) {
        SgArrayType *arrayType = isSgArrayType(variableType);
        assert(arrayType);

        SgType *elementType = arrayType->get_base_type()->findBaseType();

        TypeStruct arrayTypeDesc = TypeStruct(elementType->unparseToString());
        arrayTypeDesc.overrideType(variable->get_type()->unparseToString());
        LocalArrayNode *node = new LocalArrayNode(variable->get_name(), arrayTypeDesc);


        variableToReadNode[variable] = node;
        variableToWriteNode[variable] = node;

        addArrayPragmas(variable->get_name(), node);

        if (elementType->variantT() == V_SgClassType) {
            addStructTypeToMap(elementType);
        } else {
            if(arrayType){
                setNumElements(node, arrayType);
            }
        }
        node->setType(elementType->unparseToString());
        return node;
    } else {
        std::string description = variable->get_name();
        Node *node = new LocalScalarNode(description);
        variableToReadNode[variable] = node;
        variableToWriteNode[variable] = node;
        node->setType(variableType->findBaseType()->unparseToString());
        return node;
    }
}

Node *VariableMapper::createParameterVariableNode(SgInitializedName *variable) {
    std::string variableName = variable->get_name();

    SgType *variableType = variable->get_type();
    SgType *baseType = variableType->findBaseType();

    bool arrayOrPointer = false;
    SgType *elementType;
    int numElements = 1;
    if (variableType->variantT() == V_SgPointerType) {
        SgPointerType *pointerType = isSgPointerType(variableType);
        elementType = pointerType->get_base_type()->findBaseType();
        arrayOrPointer = true;
    }

    SgArrayType *arrayType = nullptr;
    if (variableType->variantT() == V_SgArrayType) {
        arrayType = isSgArrayType(variableType);

        elementType = arrayType->get_base_type()->findBaseType();
        arrayOrPointer = true;
    }

    if (arrayOrPointer) {
        Node *node;

        if (!finishedMain) {
            // pass the real type so that
            // if not one-hot-encoding the type we can get int/float and bitwidth
            TypeStruct arrayTypeDesc = TypeStruct(elementType->unparseToString());
            arrayTypeDesc.overrideType(variable->get_type()->unparseToString());
            node = new ExternalArrayNode(variableName, arrayTypeDesc);
        } else {
            // programl didn't string encode array type for parameters and so neither did I
            // I probably should have, with a switch
            // but since not one-hot-encoding anymore, not going to implement it
            node = new SubParameterArrayNode(variableName);
        }
        variableToReadNode[variable] = node;
        variableToWriteNode[variable] = node;

        addArrayPragmas(variable->get_name(), node);

        if (elementType->variantT() == V_SgClassType) {
            addStructTypeToMap(elementType);
        } else {
            if(arrayType){
                setNumElements(node, arrayType);
            }
            node->setType(elementType->unparseToString());
        }

        return node;
    } else if (baseType->variantT() == V_SgClassType) {
        Node *node = new StructNode();
        variableToReadNode[variable] = node;
        variableToWriteNode[variable] = node;
        nonReadVariables.insert(node);

        addStructTypeToMap(baseType);
        return node;
    } else {
        ParameterScalarNode *parameterScalar = new ParameterScalarNode(variableName);
        variableToReadNode[variable] = parameterScalar;
        variableToWriteNode[variable] = parameterScalar;
        parameterScalar->setType(variableType->findBaseType()->unparseToString());
        return parameterScalar;
    }
}
void VariableMapper::setUnderlyingVariable(SgInitializedName *variable, SgInitializedName *underlyingVariable) {
    underlyingVariableMap[variable] = underlyingVariable;
}

void VariableMapper::removeUnderlyingVariable(SgInitializedName *variable) {
    if (underlyingVariableMap.count(variable) > 0) {
        underlyingVariableMap.erase(variable);
    }
}

void VariableMapper::setParamNode(SgInitializedName *param, SgExpression *input) {
    if (SgVarRefExp *varRef = isSgVarRefExp(input)) {
        SgInitializedName *varDec = varRef->get_symbol()->get_declaration();
        SgInitializedName *underlyingVariable = getUnderlyingVariable(varDec);

        setUnderlyingVariable(param, underlyingVariable);
    } else {
        removeUnderlyingVariable(param);

        variableToReadNode[param] = graphGenerator->astParser->readExpression(input);
        nonReadVariables.insert(variableToReadNode[param]);

        if (variableToWriteNode.count(param) > 0) {
            variableToWriteNode.erase(param);
        }
    }
}

// read and write being separate is no longer necessary,
// I would be happy to have a single function to access
// the variable node
// there are some areas in the structure where a node can be read but not written
// but I don't think it matters anymore
Node *VariableMapper::writeVariable(SgInitializedName *variable) {
    SgInitializedName *underlyingVariable = getUnderlyingVariable(variable);
    if (variableToWriteNode.count(underlyingVariable)) {
        return variableToWriteNode[underlyingVariable];
    } else {
        throw std::runtime_error("Variable does not have write node: " + variable->unparseToString());
    }
}

// read and write being separate is no longer necessary,
// I would be happy to have a single function to access
// the variable node
// there are some areas in the structure where a node can be read but not written
// but I don't think it matters anymore
Node *VariableMapper::readVariable(SgInitializedName *variable) {
    SgInitializedName *underlyingVariable = getUnderlyingVariable(variable);
    if (variableToReadNode.count(underlyingVariable)) {
        return variableToReadNode[underlyingVariable];
    } else {
        // global variables are only added to the graph if they are used
        // this is the code for const arrays and pointers, without examples I'm not
        // going to try support other combinations

        // but they would be quick to implement with an example
        SgType *variableType = variable->get_type();
        size_t isConst = variableType->unparseToString().find("const");
        if (isConst != std::string::npos) {
            if (variableType->variantT() == V_SgArrayType || variableType->variantT() == V_SgPointerType) {
                TypeStruct typeDesc = TypeStruct();
                typeDesc.overrideType(variable->get_type()->unparseToString());

                // SgArrayType *arrayType = isSgArrayType(variableType);
                SgType *elementType = variableType->findBaseType();

                TypeStruct elementTypeDesc = TypeStruct(elementType->unparseToString());
                Node *constant = new GlobalArrayNode(variable->unparseToString(), elementTypeDesc, typeDesc);
                variableToReadNode[variable] = constant;
                nonReadVariables.insert(constant);
                return constant;
            }
        }
        throw std::runtime_error("Variable does not have read node: " + variable->unparseToString());
    }
}

SgInitializedName *VariableMapper::getUnderlyingVariable(SgInitializedName *variable) {
    if (underlyingVariableMap.count(variable) > 0) {
        return underlyingVariableMap[variable];
    } else {
        underlyingVariableMap[variable] = variable;
        return variable;
    }
}

void VariableMapper::addStructTypeToMap(SgType *structType) {
    if (!structFieldMap.count(structType)) {
        std::map<SgInitializedName *, StructFieldNode *> fieldsToFieldNodeMap;

        SgClassType *classType = isSgClassType(structType);
        assert(classType);
        SgDeclarationStatement *classDecStatement = classType->get_declaration()->get_definingDeclaration();
        assert(classDecStatement);
        SgClassDeclaration *classDec = isSgClassDeclaration(classDecStatement);
        assert(classDec);

        int index = 0;
        for (auto decStatement : classDec->get_definition()->get_members()) {
            if (auto varDecStatement = isSgVariableDeclaration(decStatement)) {
                for (auto varDec : varDecStatement->get_variables()) {
                    StructFieldNode *structField;
                    if (varDec->get_type()->variantT() == V_SgArrayType) {
                        // keep array type names in full
                        TypeStruct arrayType = TypeStruct();
                        arrayType.overrideType(varDec->get_type()->unparseToString());

                        structField = new StructArrayFieldNode(index, arrayType);
                    } else {
                        structField = new StructFieldNode(index);
                    }
                    structField->setType(varDec->get_type()->findBaseType()->unparseToString());

                    fieldsToFieldNodeMap[varDec] = structField;
                    index++;
                }
            }
        }

        structFieldMap[structType] = fieldsToFieldNodeMap;
    }
}

StructFieldNode *VariableMapper::getStructField(SgType *structType, SgInitializedName *variable) {
    if (structFieldMap.count(structType)) {
        std::map<SgInitializedName *, StructFieldNode *> fieldMap = structFieldMap[structType];

        if (fieldMap.count(variable)) {
            return fieldMap[variable];
        } else {
            throw std::runtime_error("Tried to access a struct field but the variable declaration wasn't found");
        }

    } else {
        throw std::runtime_error("Tried to access struct field for a struct not in the map");
    }
}

} // namespace Balor
