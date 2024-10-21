#ifndef AIR_EXPLORATION_H
#define AIR_EXPLORATION_H

#include "rose.h"
#include <unordered_set>

namespace Balor {


SgFunctionDefinition *getTopLevelFunctionDef(SgProject *project, std::string topLevelFunctionName);

SgFunctionDeclaration *getFuncDecFromCall(SgFunctionCallExp *functionCall);
bool expressionAccessesVariable(SgExpression *expression, SgInitializedName *variable);
std::vector<SgVarRefExp *> findVariableReferences(SgFunctionCallExp *functionCall, SgInitializedName *parameter);
SgInitializedName *findParameterInScope(SgFunctionCallExp *functionCall, SgInitializedName *parameter);


} // namespace Balor
#endif