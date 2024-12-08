
#ifndef AIR_COMMANDLINE_H
#define AIR_COMMANDLINE_H

#include <Rose/CommandLine.h>

#include <vector>

#include "rose.h"

namespace Balor {
namespace CommandLine {

// Use the Sawyer library that comes with Rose to parse CLI inputs
Sawyer::CommandLine::ParserResult parseCommandLine(int argc, char *argv[]);

// Some CLI args are for the AIR tool and some are for the Rose frontend
// Take only the args for the Rose frontend
std::vector<std::string> getFrontendArgs(Sawyer::CommandLine::ParserResult parserResult);

// Extract the "top" argument, the top level function of the kernel
std::string getTopLevelFunctionName(Sawyer::CommandLine::ParserResult parserResult);

// Extract where to save output files to
std::string getOutputsFolder(Sawyer::CommandLine::ParserResult parserResult);

} // namespace CommandLine
} // namespace Balor

#endif