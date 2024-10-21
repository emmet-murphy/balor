#include "nodeUtils.h"

namespace Balor {
namespace Utils {
// All comparison operations are instantiated as an integer comparison
// so they are considered a single type of operation
bool isComparisonOp(VariantT variant) {
    switch (variant) {
    case V_SgLessThanOp:
        return true;
    case V_SgLessOrEqualOp:
        return true;
    case V_SgEqualityOp:
        return true;
    case V_SgNotEqualOp:
        return true;
    case V_SgGreaterOrEqualOp:
        return true;
    case V_SgGreaterThanOp:
        return true;
    }

    return false;
}

// (X)assign ops are all handled the same way
// Though they instantiate different arithmetic nodes
// They all also add a write node
bool isUpdateOp(VariantT variant) {
    switch (variant) {
    case V_SgPlusAssignOp:
        return true;
    case V_SgMinusAssignOp:
        return true;
    case V_SgMultAssignOp:
        return true;
    case V_SgDivAssignOp:
        return true;
    case V_SgIorAssignOp:
        return true;
    case V_SgXorAssignOp:
        return true;
    case V_SgRshiftAssignOp:
        return true;
    }

    return false;
}

// (X)assign ops are all handled the same way
// Though they instantiate different arithmetic nodes
// They all also add a write node
bool isBinaryArithmeticNode(VariantT variant) {
    switch (variant) {
    case V_SgAddOp:
        return true;
    case V_SgSubtractOp:
        return true;
    case V_SgMultiplyOp:
        return true;
    case V_SgDivideOp:
        return true;
    case V_SgLshiftOp:
        return true;
    case V_SgRshiftOp:
        return true;
    case V_SgBitAndOp:
        return true;
    case V_SgBitXorOp:
        return true;
    case V_SgModOp:
        return true;
    case V_SgAndOp:
        return true;
    }

    return false;
}

// Increment and Decrement ops are handled the same way
// but instantiate different arithmetic nodes
bool isIncOrDecOp(VariantT variant) {
    switch (variant) {
    case V_SgPlusPlusOp:
        return true;
    case V_SgMinusMinusOp:
        return true;
    }

    return false;
}

// if you have a AST node and you know it is partly
// or wholely made up of an arithmetic node,
// get the arithmetic node type.
std::string getArithmeticNodeEncoding(VariantT variant) {
    if (isComparisonOp(variant)) {
        return "Comparison";
    }

    switch (variant) {
    case V_SgAddOp:
        return "Addition";
    case V_SgSubtractOp:
        return "Subtraction";
    case V_SgMultiplyOp:
        return "Multiplication";
    case V_SgDivideOp:
        return "Division";
    case V_SgPlusAssignOp:
        return "Addition";
    case V_SgMinusAssignOp:
        return "Subtraction";
    case V_SgMultAssignOp:
        return "Multiplication";
    case V_SgDivAssignOp:
        return "Division";
    case V_SgPlusPlusOp:
        return "Addition";
    case V_SgMinusMinusOp:
        return "Subtraction";
    case V_SgNotOp:
        return "Not";
    case V_SgLshiftOp:
        return "LeftShift";
    case V_SgRshiftOp:
        return "RightShift";
    case V_SgRshiftAssignOp:
        return "RightShift";
    case V_SgBitAndOp:
        return "BitAnd";
    case V_SgBitXorOp:
        return "BitXor";
    case V_SgIorAssignOp:
        return "Or";
    case V_SgXorAssignOp:
        return "Xor";
    case V_SgModOp:
        return "Mod";
    case V_SgAndOp:
        return "And";
    }

    throw std::runtime_error("Unsupported arithmetic operation: ");
}

} // namespace Utils

} // namespace Balor
