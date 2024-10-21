#include "utility.h"
#include "rose.h"
#include "unordered_set"

#include <boost/algorithm/string.hpp>


namespace Balor {

// Expressions can nest endlessly,
// so check if an expression accesses a variable at any point in its tree
bool expressionAccessesVariable(SgExpression *expression, SgInitializedName *variable) {

    // get all of the variables references in the expression
    std::vector<SgNode *> variableRefs = NodeQuery::querySubTree(expression, V_SgVarRefExp);

    // foreach variable reference
    for (SgNode *variableRefNode : variableRefs) {
        // cast to VarRefExp to get access to member variables
        SgVarRefExp *varRef = isSgVarRefExp(variableRefNode);

        // get the variable declaration
        SgInitializedName *decFromSymbol = varRef->get_symbol()->get_declaration();

        // check if this variable reference uses the variable we care about
        if (decFromSymbol == variable) {
            return true;
        }
    }
    return false;
}

// If you have the variable passed to a function call
// Find that variable inside the scope of the function definition
//
// Done by matching position in parameter list
// e.g. if the variable is the third parameter,
// get the third parameter declaration
// which I think is reliable?
//
// Limitation: no safety around if the parameter is an expression
SgInitializedName *findParameterInScope(SgFunctionCallExp *functionCall, SgInitializedName *parameter) {
    int paramIndex = 0;
    bool found = false;
    for (SgExpression *expression : functionCall->get_args()->get_expressions()) {
        // you can't directly check which expression is the variable reference
        // as they can nest.
        if (expressionAccessesVariable(expression, parameter)) {
            found = true;
            break;
        }
        paramIndex++;
    }

    if (!found) {
        std::string errorMessage = "Tried to find parameter " + parameter->unparseToString() + " in scope " +
                                   functionCall->unparseToString() + ", but it wasn't passed as an argument";
        throw std::runtime_error(errorMessage);
    }

    // get the function declaration
    SgFunctionDeclaration *functionDec = getFuncDecFromCall(functionCall);

    // get the parameter by index from the declaration args
    SgInitializedName *scopedParameter = functionDec->get_args().at(paramIndex);

    return scopedParameter;
}

// Find the SgFunctionDefinition pointer to the top level function
// specified by the CLI argument
SgFunctionDefinition *getTopLevelFunctionDef(SgProject *project, std::string topLevelFunctionName) {
    // get all function definitions in the project
    std::vector<SgNode *> functionCallList = NodeQuery::querySubTree(project, V_SgFunctionDefinition);

    // declare the pointer to return
    // Initialise to null to check if the definition was actually found
    SgFunctionDefinition *topLevelFunctionDef = NULL;

    // foreach function definition
    for (SgNode *node : functionCallList) {
        // cast to SgFunctionDefinition to get access to member variables
        SgFunctionDefinition *functionDef = isSgFunctionDefinition(node);

        // the name of the function is only stored on the declaration
        // However there's a lot of compiler generated function declarations
        // So only looking at definitions in the query means less iterations
        std::string functionName = functionDef->get_declaration()->get_name();

        if (functionName == topLevelFunctionName) {
            topLevelFunctionDef = functionDef;
            break;
        }
    }
    if (topLevelFunctionDef == NULL) {
        throw std::invalid_argument("The provided top level function \"" + topLevelFunctionName + "\" was not found.");
    }

    return topLevelFunctionDef;
}

// Get all the references to a variable inside a function call
std::vector<SgVarRefExp *> findVariableReferences(SgFunctionCallExp *functionCall, SgInitializedName *parameter) {

    // get the function declaration
    SgFunctionDeclaration *functionDec = getFuncDecFromCall(functionCall);

    // get all variable uses in the function
    std::vector<SgNode *> variableRefs = NodeQuery::querySubTree(functionDec->get_definition(), V_SgVarRefExp);

    std::vector<SgVarRefExp *> parameterUses;

    for (SgNode *varRefNode : variableRefs) {
        // get the variable refernce
        SgVarRefExp *varRef = isSgVarRefExp(varRefNode);

        // get the variable declaration
        SgInitializedName *usedVariable = varRef->get_symbol()->get_declaration();

        // if this reference is to the chosen parameter
        // add it to the vector
        if (parameter == usedVariable) {
            parameterUses.push_back(varRef);
        }
    }

    return parameterUses;
}

// Helper function which gets the defining function declaration
// from the function call
SgFunctionDeclaration *getFuncDecFromCall(SgFunctionCallExp *functionCall) {
    // get the function reference as an expression
    SgExpression *expression = functionCall->get_function();
    // convert to a function reference
    if (SgFunctionRefExp *ref = isSgFunctionRefExp(expression)) {
        // get the defining function declaration
        // going through the symbol gets you a non-definining (forward) declaration
        // even is one doesn't exist in the source code
        SgDeclarationStatement *definingDecStatement = ref->get_symbol()->get_declaration()->get_definingDeclaration();

        if (definingDecStatement) {
            // return as a function declaration
            return isSgFunctionDeclaration(definingDecStatement);
        }

        return ref->get_symbol()->get_declaration();
    } else if (SgDotExp *dot = isSgDotExp(expression)) {
        // get the defining function declaration
        // going through the symbol gets you a non-definining (forward) declaration
        // even is one doesn't exist in the source code
        if (SgMemberFunctionRefExp *ref = isSgMemberFunctionRefExp(dot->get_rhs_operand())) {
            SgDeclarationStatement *definingDecStatement =
                ref->get_symbol()->get_declaration()->get_definingDeclaration();
            // return as a function declaration
            return isSgFunctionDeclaration(definingDecStatement);
        } else {
            throw std::runtime_error("rhs of dot is not a member function ref");
        }

    } else {
        throw std::runtime_error("SgFunctionCallExp->get_function() returned something "
                                 "other than a function reference: " +
                                 expression->unparseToString());
    }
}

} // namespace Balor