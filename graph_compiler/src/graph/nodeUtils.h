#include "rose.h"

#ifndef BALOR_NODE_UTILS_H
#define BALOR_NODE_UTILS_H

namespace Balor {
namespace Utils {

// All comparison operations are instantiated as an integer comparison
// so they are considered a single type of operation
bool isComparisonOp(VariantT variant);

// (X)assign ops are all handled the same way
// Though they instantiate different arithmetic nodes
// They all also add a write node
bool isUpdateOp(VariantT variant);
// (X)assign ops are all handled the same way
// Though they instantiate different arithmetic nodes
// They all also add a write node
bool isBinaryArithmeticNode(VariantT variant);

// Increment and Decrement ops are handled the same way
// but instantiate different arithmetic nodes
bool isIncOrDecOp(VariantT variant);

// if you have a AST node and you know it is partly
// or wholely made up of an arithmetic node,
// get the arithmetic node type.
std::string getArithmeticNodeEncoding(VariantT variant);
} // namespace Utils

} // namespace Balor

#endif