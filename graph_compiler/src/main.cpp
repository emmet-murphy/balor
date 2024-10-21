#include <algorithm>
#include <functional>
#include <numeric>

#include "commandLine.h"
#include "utility.h"
#include "graph/args.h"
#include "graph/graphGenerator.h"
#include "rose.h"

int main(int argc, char *argv[]) {
    // Initialize and check compatibility. See Rose::initialize
    ROSE_INITIALIZE;

    Sawyer::CommandLine::ParserResult parserResult = Balor::CommandLine::parseCommandLine(argc, argv);

    std::vector<std::string> frontendArgs;
    std::string topLevelFunctionName;

    SgProject *project;
    SgFunctionDefinition *topLevelFunctionDef;

    try {
        frontendArgs = Balor::CommandLine::getFrontendArgs(parserResult);

        // Build the AST used by ROSE
        project = frontend(frontendArgs);
        ROSE_ASSERT(project != NULL);

        SgGlobal *globalScope = SageInterface::getFirstGlobalScope(project);
        SageBuilder::pushScopeStack(isSgScopeStatement(globalScope));

        topLevelFunctionName = Balor::CommandLine::getTopLevelFunctionName(parserResult);
        topLevelFunctionDef = Balor::getTopLevelFunctionDef(project, topLevelFunctionName);
    } catch (std::invalid_argument e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    bool makePdf = parserResult.have(Balor::MAKE_PDF);
    bool makeDot = parserResult.have(Balor::MAKE_DOT);

    Balor::GraphGenerator graphGen = Balor::GraphGenerator(parserResult);
    graphGen.generateGraph(topLevelFunctionDef);

    if (makePdf || makeDot) {
        std::string fileName = "outputs/" + topLevelFunctionName;

        std::ofstream out(fileName + ".dot");
        std::streambuf *coutbuf = std::cout.rdbuf(); // save old buf
        std::cout.rdbuf(out.rdbuf());                // redirect std::cout

        graphGen.printGraph();

        std::cout.rdbuf(coutbuf); // restore cout

        if (makePdf) {
            std::string reorderCall =
                "python scripts/reorderNodes.py " + fileName + ".dot " + fileName + "_reordered.dot";
            system(reorderCall.c_str());

            std::string dotCall = "dot -Tpdf " + fileName + "_reordered.dot -o " + fileName + ".pdf";
            system(dotCall.c_str());
        }
    } else {
        graphGen.printGraph();
    }

    return 0;
}
