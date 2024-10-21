
import re
class Directive:
    def __init__(self, directive):
        self.unrollFactor = 1
        self.arrayPartitionFactor = 1
        self.setResource = False
        self.partition = False
        self.added = False
        self.pipeline = False
        self.inline = False
        self.not_inline = False
        self.dim = 1

        directive_words = directive.split(" ")
        if "set_directive_unroll" in directive:
            index = directive_words.index("-factor")
            self.unrollFactor = int(directive_words[index+1])

            loop_labels = directive_words[index+2]
            loop_labels = loop_labels.replace('"', "")
            loop_labels = loop_labels.split('/')
            self.unrollLoopLabel = loop_labels[-1]
        elif "set_directive_array_partition" in directive:
            self.partition = True
            index = directive_words.index("-type")
            partitionType = directive_words[index + 1]
            self.partitionType = partitionType
            if partitionType == "cyclic" or partitionType == "block":
                factorIndex = directive_words.index("-factor")
                self.arrayPartitionFactor = directive_words[factorIndex + 1]
            elif partitionType == "complete":
                pass
            else:
                assert(False)

            if "dim" in directive:
                index = directive_words.index("-dim")
                self.dim = directive_words[index + 1]
            

            self.variableToPartition = directive_words[-1]
        elif "set_directive_resource" in directive:
            self.setResource = True
            self.variable = directive_words[-1]
            coreIndex = directive_words.index("-core")
            self.core = directive_words[coreIndex + 1]
        elif "set_directive_pipeline" in directive:
            self.pipeline = True
            loop_labels = directive_words[-1]
            loop_labels = loop_labels.replace('"', "")
            loop_labels = loop_labels.split('/')
            self.pipelineLoopLabel = loop_labels[-1]
        elif "set_directive_inline" in directive:
            if "-off" in directive:
                self.not_inline = True
                self.inlined_function = directive_words[-1]
            else:
                self.inline = True
                self.inlined_function = directive_words[-1]


    def pre_apply(self, codeLine):
        match = re.search("^(\s+)",codeLine)
        if match:
            indentation = match.group(1)
        else:
            indentation = ""
        if self.inline and not self.added:
            if f"{self.inlined_function}(" in codeLine:
                codeLine += indentation + "#pragma HLS inline on \n"
                self.added = True
        if self.not_inline and not self.added:
            if f"{self.inlined_function}(" in codeLine:
                codeLine += indentation + "#pragma HLS inline off \n"
                self.added = True
        return codeLine

    def apply(self, codeLine):
        match = re.search("^(\s+)",codeLine)
        if match:
            indentation = match.group(1)
        else:
            indentation = ""
        if self.unrollFactor > 1:
            if self.unrollLoopLabel + ":" in codeLine:
                # codeLine = codeLine.replace(self.unrollLoopLabel + ":", "")

                codeLine += indentation + "#pragma HLS UNROLL factor=" + str(self.unrollFactor) + "\n"
                return codeLine
        if self.partition:
            searchCodeLine = codeLine.replace("*", "")
            asLastParam = (self.variableToPartition + ")") in searchCodeLine
            asParam = (self.variableToPartition + ",") in searchCodeLine
            asDef = (self.variableToPartition + "=") in searchCodeLine
            asDef = asDef or (self.variableToPartition + " =") in searchCodeLine
            asDec = (self.variableToPartition + ";") in searchCodeLine
            asArray = (self.variableToPartition + "[") in searchCodeLine
            
            if (asLastParam or asParam or asDef or asDec or asArray) and not self.added:
                self.added = True
                pragma = f"#pragma HLS ARRAY_PARTITION type={self.partitionType} variable={self.variableToPartition} factor={self.arrayPartitionFactor} dim={self.dim}"
                
                codeLine += indentation + pragma + "\n"
                return codeLine
        if self.setResource:
            searchCodeLine = codeLine.replace("*", "")
            asLastParam = (self.variable + ")") in searchCodeLine
            asParam = (self.variable + ",") in searchCodeLine
            asDef = (self.variable + "=") in searchCodeLine
            asDef = asDef or (self.variable + " =") in searchCodeLine
            asDec = (self.variable + ";") in searchCodeLine
            asArray = (self.variable + "[") in searchCodeLine
            if (asLastParam or asParam or asDef or asDec or asArray) and not self.added:
                self.added = True
                pragma = f"#pragma HLS RESOURCE core={self.core} variable={self.variable}"
                codeLine += indentation + pragma + "\n"
        if self.pipeline:
            if self.pipelineLoopLabel + ":" in codeLine:
                codeLine += indentation + "#pragma HLS PIPELINE \n"

        return codeLine
        

class MerlinDirective():
    def __init__(self, key, value):
        self.key = key
        self.value = value
    
    def match(self, line):
        return (self.key + "}") in line
    
    def apply(self, line):
        pattern = rf"auto\{{{re.escape(self.key)}\}}"
        output = re.sub(pattern, str(self.value), line)

        return output


def apply_vitis_directives(kernel_data, i, output):
    directive_list = []

    if i != -1:
        directive_string_list = kernel_data.get_pragmas(i).split("\n")
        for directive_string in directive_string_list:
            directive = Directive(directive_string)
            directive_list.append(directive)    
        
    
    alteredCodeFile = []
    reachedKernel = False
    with open(kernel_data.input_file, 'r') as file:
        for line in file:
            if kernel_data.kernel_name in line:
                reachedKernel = True
            if reachedKernel:
                for directive in directive_list:
                    line = directive.apply(line)
                if "for" in line or "while" in line:
                    line = re.sub(r'\w+:', "", line)
            else:
                for directive in directive_list:
                    line = directive.pre_apply(line)
            alteredCodeFile.append(line)


    with open(output, 'w') as file:
        for codeLine in alteredCodeFile:
            file.write(f"{codeLine}")

def apply_merlin_directives(kernel_data, i, output):
    directive_list = []


    if i != -1:
        directive_dict = kernel_data.get_pragmas(i)

        for key in directive_dict:
            directive = MerlinDirective(key, directive_dict[key])
            directive_list.append(directive)    
    else:
        kernel_data.get_pragmas(0)

    alteredCodeFile = []
    with open(kernel_data.input_file, 'r') as file:
        for line in file:
            if "#pragma ACCEL" in line:
                for directive in directive_list:
                    if directive.match(line):
                        alteredCodeFile.append(directive.apply(line))
            else:
                alteredCodeFile.append(line)

    with open(output, 'w') as file:
        for codeLine in alteredCodeFile:
            file.write(f"{codeLine}")