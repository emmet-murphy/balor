#ifndef BALOR_EDGE_PRINTER_H
#define BALOR_EDGE_PRINTER_H

#include <map>
#include <string>

namespace Balor {
class EdgePrinter {
  public:
    EdgePrinter(int id1, int id2);

    int id1, id2;
    int order;

    std::map<std::string, std::string> attributes;

    void print();
};
} // namespace Balor

#endif