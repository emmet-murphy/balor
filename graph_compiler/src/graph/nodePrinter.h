#ifndef BALOR_NODE_PRINTER_H
#define BALOR_NODE_PRINTER_H

#include "node.h"
#include <map>
#include <string>

namespace Balor {

class Node;

class NodePrinter {
  public:
    NodePrinter(Node *node, const std::string &color);
    std::string color;
    std::map<std::string, std::string> attributes;

    Node *node;

    void print();
};

} // namespace Balor

#endif