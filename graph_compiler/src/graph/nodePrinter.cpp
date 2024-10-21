#include "nodePrinter.h"
#include "args.h"
#include "node.h"

namespace {
std::string addPragmaToLabel(Balor::Node *node, std::string label) {
    if (node->unrollFactor.full > 1) {
        label += "\n Unroll: " + std::to_string(node->unrollFactor.full);
    }
    if (node->tile > 1) {
        label += "\n Tile: " + std::to_string(node->tile);
    }
    if (node->partitionFactor1 > 0) {
        label += "\n Partition Factor 1: " + std::to_string(node->partitionFactor1);
    }
    if (node->partitionFactor2 > 0) {
        label += "\n Partition Factor 2: " + std::to_string(node->partitionFactor2);
    }
    if (node->partitionFactor3 > 0) {
        label += "\n Partition Factor 3: " + std::to_string(node->partitionFactor3);
    }
    if(node->tripcount.full > 1){
        label += "\n Tripcount: " + std::to_string(node->tripcount.full);
    }

    if (node->partitionType1 != "none") {
        label += "\n Partition 1: " + node->partitionType1;
    }
    if (node->partitionType2 != "none") {
        label += "\n Partition 2: " + node->partitionType2;
    }
    if (node->partitionType3 != "none") {
        label += "\n Partition 3: " + node->partitionType3;
    }
    if(node->resourceType != "none"){
        label += "\n Resource Type: " + node->resourceType;
    }
    if (node->pipelined){
        label += "\n Pipelined";
    }
    if (node->previouslyPipelined){
        label += "\n Previously Pipelined";
    }
    if(node->pipelinedType == Balor::PipelinedType::FINE){
        label += "\n Fine-grained pipelining";
    }
    if(node->pipelinedType == Balor::PipelinedType::COARSE){
        label += "\n Coarse-grained pipelining";
    }
    if(Balor::Nodes::graphGenerator->getFuncInlined(node->funcDec)){
        label += "\n Inlined";
    } else {
        label += "\n Not Inlined";
    }
    return label;
}

void output(std::string message) { 
    std::cout << message << std::endl; 
}

std::string toVariableType(Balor::Node *node) {
    Balor::TypeStruct type;
    try{
        type = node->getImmediateType();
    } catch(...){
        return "NA";
    }
    if (type.isVoid) {
        return "void";
    }
    if (type.dataType == Balor::DataType::INTEGER) {
        return "int";
    } else if (type.dataType == Balor::DataType::FLOAT) {
        return "float";
    } else if (type.dataType == Balor::DataType::STRUCT){
        return "struct";
    }
    throw std::runtime_error("toVariableType reached unreachable control flow");
}

std::string typeToBitwidth(Balor::Node *node) {
    Balor::TypeStruct type;
    try{
        type = node->getImmediateType();
    } catch(...){
        return "0";
    }
    if (type.isVoid) {
        return "0";
    }
    return std::to_string(type.bitwidth);
}

} // namespace

namespace Balor {
NodePrinter::NodePrinter(Node *node, const std::string &color) : color(color) {
    this->node = node;
    attributes["group"] = node->groupName;
    attributes["nodeType"] = "instruction";
    attributes["datasetIndex"] = node->datasetIndex;
    attributes["graphType"] = node->graphType;

    // if(Nodes::graphGenerator->checkArg())

    if (Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        attributes["unrollFactor1"] = std::to_string(node->unrollFactor.first);
        attributes["unrollFactor2"] = std::to_string(node->unrollFactor.second);
        attributes["unrollFactor3"] = std::to_string(node->unrollFactor.third);
        attributes["fullUnrollFactor"] = std::to_string(node->unrollFactor.full);
        attributes["tile"] = std::to_string(node->tile);
        attributes["partitionFactor1"] = std::to_string(node->partitionFactor1);
        attributes["partitionFactor2"] = std::to_string(node->partitionFactor2);
        attributes["partitionFactor3"] = std::to_string(node->partitionFactor3);
        attributes["partition1"] = node->partitionType1;
        attributes["partition2"] = node->partitionType2;
        attributes["partition3"] = node->partitionType3;
        attributes["inlined"] = std::to_string(Nodes::graphGenerator->getFuncInlined(node->funcDec));
        attributes["resourceType"] = node->resourceType;
        attributes["tripcount"] = std::to_string(node->tripcount.full);
        attributes["pipelined"] = std::to_string(node->pipelined);

        // this shouldn't be necessary? but got weird bug
        std::string previouslyPipelined = node->previouslyPipelined ? "1" : "0";
        attributes["previouslyPipelined"] = previouslyPipelined;


        attributes["pipelinedType"] = std::to_string(int(node->pipelinedType));
    } else {
        attributes["numeric"] = "0";
    }

    if (Nodes::graphGenerator->checkArg(ABSORB_TYPES)) {
        if (!Nodes::graphGenerator->checkArg(DONT_DISPLAY_TYPES)) {
            if (Nodes::graphGenerator->checkArg(ONE_HOT_TYPES)) {
                attributes["datatype"] = node->getTypeToPrint();
            } else {
                attributes["datatype"] = toVariableType(node);
                attributes["bitwidth"] = typeToBitwidth(node);
                attributes["totalArrayWidth"] = std::to_string(node->totalNumElements);
                attributes["arrayWidth0"] = std::to_string(node->numElements0);
                attributes["arrayWidth1"] = std::to_string(node->numElements1);
                attributes["arrayWidth2"] = std::to_string(node->numElements2);
                attributes["arrayWidth3"] = std::to_string(node->numElements3);
                attributes["arrayWidth4"] = std::to_string(node->numElements4);
            }
        }
    }

    if (Nodes::graphGenerator->checkArg(ADD_BB_ID)) {
        attributes["bbID"] = std::to_string(node->bbID);
    }
    if (Nodes::graphGenerator->checkArg(ADD_FUNC_ID)) {
        attributes["funcID"] = std::to_string(node->functionID);
    }
    if(Nodes::graphGenerator->checkArg(ADD_NUM_CALLS)){
        assert(Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS));
        if(Nodes::graphGenerator->getFuncInlined(node->funcDec)){
            attributes["numCalls"] = std::to_string(Nodes::graphGenerator->getCallsNums(node->funcDec));
            attributes["numCallSites"] = std::to_string(Nodes::graphGenerator->getCallSiteNums(node->funcDec));
            // attributes["inlined"] = "inlined";
        }
        else {
            attributes["numCalls"] = std::to_string(1);
            attributes["numCallSites"] = std::to_string(1);
        }
    }
}

void NodePrinter::print() {
    std::string out = "node" + std::to_string(node->id);
    out += " [";
    out += "style=filled fillcolor=\"" + color + "\" ";

    if (Nodes::graphGenerator->checkArg(ABSORB_PRAGMAS)) {
        attributes["label"] = addPragmaToLabel(node, attributes["label"]);
    }

    if (attributes.count("datatype")) {
        attributes["label"] += "\n" + attributes["datatype"];
    }
    if (attributes.count("bitwidth")) {
        attributes["label"] += "\n" + attributes["bitwidth"] + " bits";
    }
    if (attributes.count("totalArrayWidth") && attributes["totalArrayWidth"] != "1") {
        attributes["label"] += "\n Total Array Width: " + attributes["totalArrayWidth"];
    }
    if (attributes.count("arrayWidth0") && attributes["arrayWidth0"] != "1") {
        attributes["label"] += "\n Array Width 0: " + attributes["arrayWidth0"];
    }
    if (attributes.count("arrayWidth1") && attributes["arrayWidth1"] != "1") {
        attributes["label"] += "\n Array Width 1: " + attributes["arrayWidth1"];
    }
    if (attributes.count("arrayWidth2") && attributes["arrayWidth2"] != "1") {
        attributes["label"] += "\n Array Width 2: " + attributes["arrayWidth2"];
    }
    if (attributes.count("arrayWidth3") && attributes["arrayWidth3"] != "1") {
        attributes["label"] += "\n Array Width 3:" + attributes["arrayWidth3"];
    }
    if (attributes.count("arrayWidth4") && attributes["arrayWidth4"] != "1") {
        attributes["label"] += "\n Array Width 4:" + attributes["arrayWidth4"];
    }
    if (attributes.count("bbID")) {
        attributes["label"] += "\n BB ID: " + attributes["bbID"];
    }
    if (attributes.count("funcID")) {
        attributes["label"] += "\n Func ID: " + attributes["funcID"];
    }

    if (!Nodes::graphGenerator->checkArg(ADD_NODE_TYPE)) {
        attributes.erase("nodeType");
    }

    if (attributes.count("nodeType")) {
        attributes["label"] += "\n Node Type: " + attributes["nodeType"];
    }

    if(attributes.count("numCalls")){
        attributes["label"] += "\n Num Calls: " + attributes["numCalls"];
    }

    if(attributes.count("numCallSites")){
        attributes["label"] += "\n Num Call Sites: " + attributes["numCallSites"];
    }


    if (!node->extraNote.empty()) {
        attributes["label"] += "\n Extra note " + node->extraNote;
    }

    // attributes["label"] += "\n " + node->datasetIndex;

    for (std::pair<std::string, std::string> attribute : attributes) {
        out += attribute.first;
        out += "=\"";
        out += attribute.second;
        out += "\" ";
    }
    out += "]";
    output(out);
}
} // namespace Balor