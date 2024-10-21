#include "pragmaParser.h"
#include "rose.h"

#include <boost/algorithm/string.hpp>

namespace Balor {

void PragmaParser::parseInlinePragmas(std::set<SgFunctionDeclaration *> funcDecs) {
    for(SgFunctionDeclaration *funcDec : funcDecs){
        if (parseInlinePragma(funcDec)) {
            inlinedFunctions.push(funcDec);
        }
    }
}

bool PragmaParser::parseInlinePragma(SgFunctionDeclaration *funcDec) {
    if (funcDec->get_definition()) {
        SgBasicBlock *bb = funcDec->get_definition()->get_body();

        functionInlined = false;
        parsePragmas(bb);

        return functionInlined;
    }
    return false;
}

void PragmaParser::parsePragmas(SgBasicBlock *bb) {
    std::vector<SgNode *> pragmas = NodeQuery::querySubTree(bb, V_SgPragmaDeclaration, AstQueryNamespace::ChildrenOnly);

    int unrollFactor = 1;
    float tripcount = 1;
    int tile = 1;
    bool pipelined = false;
    PipelinedType pipelinedType = PipelinedType::NOT;

    for (SgNode *pragmaNode : pragmas) {
        std::cerr << pragmaNode->unparseToString() << std::endl;
        // cast to SgPragma to get access to member variables
        SgPragma *pragma = isSgPragmaDeclaration(pragmaNode)->get_pragma();

        std::string pragmaText = pragma->get_name();
        // convert the pragma text (everything after #pragma) to uppercase
        std::string pragmaTextUpper = boost::algorithm::to_upper_copy(pragma->get_name());

        // split the pragma into word tokens using boost to prevent whitespace issues
        std::vector<std::string> pragmaTextVectorUpper;
        boost::algorithm::split(pragmaTextVectorUpper, pragmaTextUpper, boost::is_any_of(" ="));

        std::vector<std::string> pragmaTextVector;
        boost::algorithm::split(pragmaTextVector, pragmaText, boost::is_any_of(" ="));

        if (pragmaTextVectorUpper.size() > 1) {
            if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "UNROLL") {
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            unrollFactor = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read unroll factor from pragma");
                        }
                    }
                }
            }else if (pragmaTextVectorUpper[0] == "ACCEL" && pragmaTextVectorUpper[1] == "PARALLEL"){
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            unrollFactor = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read unroll factor from pragma");
                        }
                    }
                }
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "PIPELINE") {
                pipelined = true;
                pipelinedType = PipelinedType::FINE;
            }else if(pragmaTextVectorUpper[0] == "ACCEL" && pragmaTextVectorUpper[1] == "PIPELINE"){
                pipelined = true;
                pipelinedType = PipelinedType::COARSE;
                for (int i = 2; i < pragmaTextVectorUpper.size(); i++) {
                    if (pragmaTextVectorUpper[i] == "OFF") {
                        pipelined = false;
                        pipelinedType = PipelinedType::NOT;
                    }
                    if (pragmaTextVectorUpper[i] == "FLATTEN") {
                        pipelinedType = PipelinedType::FINE;
                    }
                }
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "RESOURCE") {
                std::string core;
                std::string variable;
                bool foundCore = false;
                bool foundVariable = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "CORE") {
                        foundCore = true;
                        core = pragmaTextVector[i + 1];
                    }
                    if (pragmaTextVectorUpper[i] == "VARIABLE") {
                        foundVariable = true;
                        variable = pragmaTextVector[i + 1];
                    }
                }
                if (!foundCore || !foundVariable) {
                    throw std::runtime_error("Couldn't find core or variable on resource pragma");
                }

                graphGenerator->variableMapper->resourceTypeMap[variable] = core;
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "ARRAY_PARTITION") {
                std::string type;
                int factor;
                int dim;
                std::string variable;
                bool foundType = false;
                bool foundVariable = false;
                bool foundFactor = false;
                bool foundDim = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "TYPE") {
                        foundType = true;
                        type = pragmaTextVector[i + 1];
                    } else if (pragmaTextVectorUpper[i] == "VARIABLE") {
                        foundVariable = true;
                        variable = pragmaTextVector[i + 1];
                    } else if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            foundFactor = true;
                            factor = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read factor from array partition pragma");
                        }
                    } else if (pragmaTextVectorUpper[i] == "DIM") {
                        try {
                            foundDim = true;
                            dim = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read dim from array partition pragma");
                        }
                    }
                }
                bool found = false;
                if (foundType && foundVariable && foundFactor && foundDim) {
                    found = true;
                } else if (foundType && foundVariable && foundDim && type == "complete") {
                    found = true;
                    factor = 1;
                } else {
                    throw std::runtime_error("Couldn't find one of type, variable, factor or dim on array partition pragma");
                }
                graphGenerator->variableMapper->arrayPartitionMap[variable].push(std::make_tuple(type, factor, dim));
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "INLINE") {
                if (pragmaTextVectorUpper.size() < 3) {
                    throw std::runtime_error("Please specify on or off for inline pragma");
                }
                if (pragmaTextVectorUpper[2] == "ON") {
                    functionInlined = true;
                }
            } else if (pragmaTextVectorUpper[0] == "HLS" && pragmaTextVectorUpper[1] == "TRIPCOUNT") {
                bool foundAvg = false;
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "AVG") {
                        try {
                            foundAvg = true;
                            tripcount = std::stof(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read average tripcount from tripcount pragma");
                        }
                    }
                }
                if (!foundAvg) {
                    throw std::runtime_error("Couldn't find avg on tripcount pragma");
                }
            } else if(pragmaTextVectorUpper[0] == "ACCEL" && pragmaTextVectorUpper[1] == "TILE"){
                for (int i = 2; i < pragmaTextVectorUpper.size() - 1; i++) {
                    if (pragmaTextVectorUpper[i] == "FACTOR") {
                        try {
                            tile = std::stoi(pragmaTextVector[i + 1]);
                        } catch (std::exception e) {
                            throw std::runtime_error("Couldn't read unroll factor from pragma");
                        }
                    }
                }
            }
        }
    }

    std::cerr << "finished pragmas" << std::endl;

    unrollHierarchy.moveDown(unrollFactor);
    tripcountHierarchy.moveDown(tripcount);
    tileHierarchy.moveDown(tile);

    // runs only the first loop nest after a pipelined loop
    // and sets flag to true
    if(!pipelineStack.empty() && pipelineStack.top()){
        previouslyPipelined = true;
    }


    pipelineStack.push(pipelined);

    if(pipelined){
        pipelineTripcount = tripcountHierarchy.fullFactor;
        currentPipelinedType = pipelinedType;
    }

}

void PragmaParser::unstackPragmas() {
    unrollHierarchy.moveUp();
    tripcountHierarchy.moveUp();
    tileHierarchy.moveUp();

    pipelineStack.pop();

    if(pipelineStack.top()){
        previouslyPipelined = false;
    }
}

void PragmaParser::enterLoopCondition() {
    unrollHierarchy.pauseFactor();
    tripcountHierarchy.pauseFactor();
    tileHierarchy.pauseFactor();
}
void PragmaParser::exitLoopCondition() {
    unrollHierarchy.unpauseFactor();
    tripcountHierarchy.unpauseFactor();
    tileHierarchy.unpauseFactor();
}

void PragmaParser::enterLoopInc() {
    unrollHierarchy.pauseFactor();
    tileHierarchy.pauseFactor();
}
void PragmaParser::exitLoopInc() {
    unrollHierarchy.unpauseFactor();
    tileHierarchy.unpauseFactor();
}

StackedFactor PragmaParser::getUnrollFactor() { 
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->unrollFactor;
    }
    return {unrollHierarchy.fullFactor,  unrollHierarchy.factor1, unrollHierarchy.factor2, unrollHierarchy.factor3}; 
}

StackedFactor PragmaParser::getTripcount(){
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->tripcount;
    }
    return {tripcountHierarchy.fullFactor, tripcountHierarchy.factor1, tripcountHierarchy.factor2, tripcountHierarchy.factor3}; 
}

int PragmaParser::getTile(){
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->tile;
    }
    return tileHierarchy.fullFactor;
}


bool PragmaParser::getPipelined() { 
    if(!pipelineStack.empty()){
        return pipelineStack.top();
    }
    return false;
}

bool PragmaParser::getPreviouslyPipelined() { 
    return previouslyPipelined;
}

float PragmaParser::getPipelineTripcount(){
    return pipelineTripcount;
}

PipelinedType PragmaParser::getPipelinedType(){
    if(graphGenerator->stateNode){
        return graphGenerator->stateNode->pipelinedType;
    }
    return currentPipelinedType;
}

} // namespace Balor

