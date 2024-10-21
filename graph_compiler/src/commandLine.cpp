#include "commandLine.h"
#include "graph/args.h"
namespace {

void addHelpArg(Sawyer::CommandLine::SwitchGroup &inputArgGroup) {
    using namespace Sawyer::CommandLine;

    // create help arg
    Switch help = Switch("help", 'h');

    // set arg to show man page
    help.action(showHelpAndExit(0));

    // specify arg description in man page
    help.doc("Show this documentation.");

    // register arg
    inputArgGroup.insert(help);
}

void addTopArg(Sawyer::CommandLine::SwitchGroup &inputArgGroup) {
    using namespace Sawyer::CommandLine;

    // create top arg
    Switch top = Switch("top");

    // specify that the top arg takes a string as argument
    // argument name is "functionName" in the man page
    top.argument("functionName", anyParser());

    // specify arg description in man page
    top.doc("Specify the top level function of the kernel.");

    // register arg
    inputArgGroup.insert(top);
}

void addSrcArg(Sawyer::CommandLine::SwitchGroup &inputArgGroup) {
    using namespace Sawyer::CommandLine;

    // create src arg
    Switch src = Switch("src");

    // specify that the src arg takes a string as argument
    // argument name is "sourceFile" in the man page
    src.argument("sourceFile", anyParser());

    // specify arg description in man page
    src.doc("Specify the source file to be analyzed.");

    // register arg
    inputArgGroup.insert(src);
}

void addDatasetIndexArg(Sawyer::CommandLine::SwitchGroup &inputArgGroup) {
    using namespace Sawyer::CommandLine;

    // create src arg
    Switch datasetNumber = Switch("datasetIndex");

    // specify that the src arg takes a string as argument
    // argument name is "sourceFile" in the man page
    datasetNumber.argument("datasetIndex", anyParser());

    // specify arg description in man page
    datasetNumber.doc("Specify the the index of the dataset, so each individual node knows the top level metrics");

    // register arg
    inputArgGroup.insert(datasetNumber);
}

void addGraphTypeArg(Sawyer::CommandLine::SwitchGroup &inputArgGroup) {
    using namespace Sawyer::CommandLine;

    // create src arg
    Switch graphType = Switch("graphType");

    // specify that the src arg takes a string as argument
    // argument name is "sourceFile" in the man page
    graphType.argument("graphType", anyParser());

    // specify arg description in man page
    graphType.doc("Specify the the graph type of the dataset (Used for Vitis HLS or Merlin Compiler currently, but is fully flexible). The graph type number is appended to each node in the graph");

    // register arg
    inputArgGroup.insert(graphType);
}

Sawyer::CommandLine::SwitchGroup specifyInputArgs() {
    using namespace Sawyer::CommandLine;

    SwitchGroup inputArgGroup;
    inputArgGroup.doc("Balor Switches:");

    addHelpArg(inputArgGroup);
    addTopArg(inputArgGroup);
    addSrcArg(inputArgGroup);
    addDatasetIndexArg(inputArgGroup);
    addGraphTypeArg(inputArgGroup);

    // add the other args
    for (auto argTuple : Balor::ARGS) {
        std::string argName = argTuple.first;
        std::string argDesc = argTuple.second;

        Switch arg = Switch(argName);
        arg.doc(argDesc);

        inputArgGroup.insert(arg);
    }

    return inputArgGroup;
}

Sawyer::CommandLine::Parser makeParser(Sawyer::CommandLine::SwitchGroup inputArgGroup) {
    using namespace Sawyer::CommandLine;

    Parser parser;

    // Specify first line of program man page
    parser.purpose("Convert behavioural C++ code to architectural representations");

    // Specify usage
    parser.doc("Synopsis", "@prop{programName} [@v{switches}]");

    // Add arg set to parser
    parser.with(inputArgGroup);

    // Specify the Sawyer parser to work with the Rose compiler
    // Args not specified here should be passed down the chain
    parser.skippingNonSwitches(true);
    parser.skippingUnknownSwitches(true);

    return parser;
}

} // namespace

namespace Balor {
namespace CommandLine {
// Use the Sawyer library that comes with Rose to parse CLI inputs
Sawyer::CommandLine::ParserResult parseCommandLine(int argc, char *argv[]) {
    using namespace Sawyer::CommandLine;

    Sawyer::CommandLine::SwitchGroup inputArgGroup = specifyInputArgs();

    Parser parser = makeParser(inputArgGroup);

    ParserResult parserResult = parser.parse(argc, argv);

    // update internal variables, run args with actions attached
    parserResult.apply();

    return parserResult;
}

// Some CLI args are for the Balor tool and some are for the input code
// Take only the args for the input code
std::vector<std::string> getFrontendArgs(Sawyer::CommandLine::ParserResult parserResult) {
    // use this if for some reason you need to pass args to gcc or rose
    std::vector<std::string> frontendArgs = parserResult.unparsedArgs();

    if (!parserResult.have("src")) {
        throw std::invalid_argument("Please specify the source file using the --src arg.");
    }

    // throwing this check here so it happens early, it doesn't really fit here
    if (!parserResult.have("datasetIndex")) {
        throw std::invalid_argument("Please specify the dataset index using the --datasetIndex arg.");
    }

    // throwing this check here so it happens early, it doesn't really fit here
    if (!parserResult.have("graphType")) {
        throw std::invalid_argument("Please specify the graph type using the --graphType arg.");
    }



    // add the compile only flag to prevent generation of an object file
    // as gcc will complain that there's no "main" function
    frontendArgs.push_back("-c");

    // and get the source file
    std::string srcFile = parserResult.parsed("src").back().asString();
    frontendArgs.push_back(srcFile);

    return frontendArgs;
}

std::string getTopLevelFunctionName(Sawyer::CommandLine::ParserResult parserResult) {
    if (!parserResult.have("top")) {
        throw std::invalid_argument("Please specify the top level function of the kernel using the --top arg.");
    }
    return parserResult.parsed("top").back().asString();
}
} // namespace CommandLine
} // namespace Balor