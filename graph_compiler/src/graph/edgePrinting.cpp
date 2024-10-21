#include "args.h"
#include "edge.h"
#include "edgePrinter.h"

namespace {

void output(const std::string &content) { std::cout << content << std::endl; }

void controlFlowEdge(int id1, int id2, bool backEdge) {
    Balor::EdgePrinter printer(id1, id2);
    printer.attributes["color"] = "red";
    if (backEdge) {
        printer.attributes["edgeOrder"] = "1";
        printer.attributes["style"] = "dashed";
        printer.attributes["dir"] = "back";
    }
    printer.attributes["flowType"] = "control";

    printer.print();
}

void callEdge(int id1, int id2, int order) {
    Balor::EdgePrinter printer(id1, id2);
    printer.attributes["color"] = "magenta";
    printer.attributes["edgeOrder"] = std::to_string(order);
    printer.attributes["flowType"] = "call";

    printer.print();
}

} // namespace

namespace Balor {

EdgePrinter::EdgePrinter(int id1, int id2) : id1(id1), id2(id2) { attributes["edgeOrder"] = "0"; }

void EdgePrinter::print() {
    std::string out = "node" + std::to_string(id1) + " -> node" + std::to_string(id2);
    out += "[";

    // Maybe would be better never to add it?
    if (!Edges::graphGenerator->checkArg(ADD_EDGE_ORDER)) {
        attributes.erase("edgeOrder");
    }

    if (attributes.count("edgeOrder")) {
        attributes["xlabel"] = attributes["edgeOrder"];
    }

    for (std::pair<std::string, std::string> attribute : attributes) {
        out += attribute.first;
        out += "=\"";
        out += attribute.second;
        out += "\" ";
    }
    out += "]";
    output(out);
}

void Edges::printSubControlFlowEdge(Node *source, Node *destination) {
    printSubControlFlowEdge(source, destination, false);
}

void Edges::printSubControlFlowEdge(Node *source, Node *destination, bool backEdge) {
    // control flow edges from the control node are actually call edges
    if (source->getVariant() == NodeVariant::EXTERNAL || destination->getVariant() == NodeVariant::EXTERNAL) {
        printSubFunctionCallEdge(source, destination, source->functionID);
    } else if (!graphGenerator->checkArg(IGNORE_CONTROL_FLOW)) {
        controlFlowEdge(source->id, destination->id, backEdge);
    }
}

void Edges::printSubFunctionCallEdge(Node *source, Node *destination) {
    printSubFunctionCallEdge(source, destination, 0);
}

void Edges::printSubFunctionCallEdge(Node *source, Node *destination, int order) {
    // put nodes with function call edges from external at the top of their subgraph
    if (source->getVariant() == NodeVariant::EXTERNAL) {
        output("subgraph cluster_" + destination->groupName + " {");
        output("{rank=min; node" + std::to_string(destination->id) + "}");
        output("}");
    }

    bool externalSource = source->getVariant() == NodeVariant::EXTERNAL;
    bool externalDestination = destination->getVariant() == NodeVariant::EXTERNAL;
    // don't add an edge to the external node
    // if we're not printing the external node
    if(externalSource || externalDestination){
        if(!Edges::graphGenerator->checkArg(ADD_EXTERNAL_NODE)){
            return;
        }
    }
    
    // if not inlining
    if (!graphGenerator->checkArg(INLINE_FUNCTIONS)) {
        // not ignore control flow
        if (!graphGenerator->checkArg(IGNORE_CONTROL_FLOW)) {
            // and not ignoring call edges
            if (!graphGenerator->checkArg(IGNORE_CALL_EDGES)) {
                // then add a call edge
                callEdge(source->id, destination->id, order);
            }
        }
    } else if (!graphGenerator->checkArg(IGNORE_CONTROL_FLOW)) {
        // else if we're inlining and not ignoring control flow
        // add a control flow edge
        controlFlowEdge(source->id, destination->id, false);
    }
}

void Edges::printSubMemoryAddressEdge(Node *source, Node *destination) {
    Balor::EdgePrinter printer(source->id, destination->id);

    // memory address edges only exist if treated variable declarations as mem elements
    if (!Edges::graphGenerator->checkArg(ALLOCAS_TO_MEM_ELEMS)) {
        printer.attributes["flowType"] = "dataflow";
        printer.attributes["color"] = "black";
    } else {
        printer.attributes["flowType"] = "address";
        printer.attributes["color"] = "aquamarine4";
    }

    printer.print();
}
void Edges::printSubDataFlowEdge(Node *source, Node *destination) { printSubDataFlowEdge(source, destination, 0); }

void Edges::printSubDataFlowEdge(Node *source, Node *destination, int order) {
    Balor::EdgePrinter printer(source->id, destination->id);
    printer.attributes["color"] = "black";
    printer.attributes["edgeOrder"] = std::to_string(order);
    printer.attributes["flowType"] = "dataflow";

    printer.print();
}

void Edges::printPragmaEdge(Node *source, Node *destination, int order) {
    Balor::EdgePrinter printer(source->id, destination->id);
    printer.attributes["color"] = "blue";
    printer.attributes["edgeOrder"] = std::to_string(order);
    printer.attributes["flowType"] = "pragma";

    printer.print();
}

} // namespace Balor