from balorgnn.generate.encoders import OneHotEncoder, NormalizedEncoder, LogNormalizedEncoder, EncoderMethod, EncoderType


baseline_nodes = ["[external]", "alloca", "store", "load", "icmp", "br", "getelementptr", "sub", "mul", "add", "fsub", "fmul", "fadd", "sext", "ret", "bitcast", "truncate", "call", "div", "sitofp", ";undefinedfunction", "fcmp", "zext", "trunc", "ashr", "and", "or", "xor", "shl", "fdiv", "phi", "fneg", "mod"]

pragma_nodes = ["resourceAllocation_bram2p", "resourceAllocation_bram1p", "cyclicArrayPartition1", "cyclicArrayPartition2", "blockArrayPartition1", "blockArrayPartition2", "completeArrayPartition1", "completeArrayPartition2", "unroll", "pipeline", "inlinedFunction"]

type_nodes = ["f32", "i64", "i32", "i1", "i8", "::prob_t[140][64]", "::prob_t[140]", "::int32_t[2048]", "::uint8_t[32]", "const::uint8_t[256]", "double[64]", "int[8]","double[3]", "double[192]", "double[4096]", "double[832]", "void", "float[64][64]", "float[64]", "double[8]", "::node_t[256]", "::edge_t[4096]", "::level_t[256]", "::edge_index_t[10]", "int[2048]", "int[128]", "double[1024]", "double[512]", "double[256]", "::int32_t[4096]", "double[13]", "::int32_t[2]", "double[4940]", "::int32_t[16384]", "::int32_t[4940]", "double[494]", "int[4096]", "::tok_t[140]", "::uint8_t*", "::prob_t[64]", "::aes256_context*", "::uint8_t[16]", "::int32_t[8192]", "::int32_t[9]", "double[2119]", "double[489]", "::prob_t[4096]", "::state_t[140]", "double[40][50]", "double[40][60]","double[60][50]", "double[50][70]", "double[50][80]", "double[80][70]", "double[40][70]", "double[70][50]", "double[60][60]", "double[40][80]", "unsignedchar[32]", 'unsignedchar[16]', "constunsignedchar[256]", "double[390][410]", "double[124][116]", "double[410][390]", "double[100][80]", "double[410]", "double[116]", "double[390]", "double[80][80]", "double[25][20][30]", "double[124]", "double[80]", "double[30][30]", "double[30][30]", "double[30]", "double[60][70]", "double[120][120]", "double[400][400]", "double[90][90]", "double[250][250]", "double[20][20][20]", "double[120]", "double[90][90]", "double[60][80]", "double[400]", "double[90]", "double[250]", "char[128]", "double[1666]", "int[4940]", "char[256]", "int[1666]", "int[8192]", "long[39304]", "double[200][240]", "double[80][60]", "int[16641]", "int[495]", "int[9]", "long[32768]", "double[200][200]", "char[16641]", "double[116][124]", "double[40]"]

baseline_edges = ["dataflow", "call", "control"]

converted_nodes = ["[external]", "ret", "externalArray", "externalScalar", "localScalar", "localArray", "globalArray", "arrayParameter", "store", "load", "cmp", "br", "sub", "mul", "getelementptr", "add", "specifyAddress", "div", "call", "sitofp", "cos", "sin", "ashr", "and", "or", "xor", "shl", "phi", "fneg", "exp", "sqrt", "pow", "buffer_fill", "buffer_empty", "direct_read", "direct_write", "mod", "read_from_stream", "write_to_stream"]
converted_edges = ["dataflow", "control", "call", "address"]

conversion_args = " --allocas_to_mem_elems --remove_sexts --remove_single_target_branches --drop_func_call_proc"

def getDatasetIndexEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "datasetIndex"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = ["0", "1", "2", "3", "4", "5", "6"]
    enc.fit()

    return enc

def getGraphTypeEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "graphType"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = ["0", "1"]
    enc.fit()

    return enc


def getKeyTextEncoder(tags):
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "keyText"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = tags
    enc.fit()

    return enc

def getNumericEncoder():
    numericEncoder = NormalizedEncoder()
    numericEncoder.type = EncoderType.NODE
    numericEncoder.label = "numeric"
    numericEncoder.method = EncoderMethod.NORMALIZED
    numericEncoder.set_max(256)
    return numericEncoder

def getUnrollFactorEncoder():
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "fullUnrollFactor"
    encoder.method = EncoderMethod.LOG_NORMALIZED
    encoder.set_max(512)
    encoder.set_bias(2)
    return encoder


def getHierarchicalUnrollFactorEncoder(index):
    encoder = NormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"unrollFactor{index+1}"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(512)
    return encoder

def getPartitionFactorEncoder(dim):
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"partitionFactor{dim}"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(512)
    encoder.set_bias(2)
    return encoder

def getTotalArrayWidthEncoder():
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"totalArrayWidth"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(160000)
    encoder.set_bias(0.9)
    return encoder

def getArrayWidthEncoder(dim):
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"arrayWidth{dim}"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(16000)
    encoder.set_bias(0.9)
    return encoder


def getTileEncoder():
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"tile"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(512)
    encoder.set_bias(2)
    return encoder

def getPipelinedTypeEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "pipelinedType"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["0", "1", "2"]
    encoder.fit()
    return encoder


def getFlowEncoder(tags):
    enc = OneHotEncoder()
    enc.type = EncoderType.EDGE
    enc.label = "flowType"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = tags
    enc.fit()

    return enc

def getBBEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "bbID"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = []
    for i in range(0, 200):
        enc.tags.append(str(i))
    enc.fit()

    return enc

def getNodeTypeEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "nodeType"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = ["instruction", "pragma", "variable", "constant"]
    enc.fit()

    return enc

def getFuncIDEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.NODE
    enc.label = "funcID"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = []
    for i in range(0, 20):
        enc.tags.append(str(i))
    enc.fit()

    return enc

def getPositionEncoder():
    enc = OneHotEncoder()
    enc.type = EncoderType.EDGE
    enc.label = "edgeOrder"
    enc.method = EncoderMethod.ONE_HOT
    enc.tags = []
    for i in range(0, 20):
        enc.tags.append(str(i))
    enc.fit()

    return enc

def getTypeEncoder(proxyPrograml, reduceIteratorBitwidth):
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "datatype"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = list(type_nodes)
    encoder.tags.append("NA")
    if proxyPrograml:
        encoder.tags.append("double")
    else:
        encoder.tags.append("f64")

    if reduceIteratorBitwidth:
        bitwidths = ["i" + str(j) for j in range(13)]
        encoder.tags += bitwidths
    encoder.fit()

    return encoder

def getPartitionEncoder(dim):
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = f"partition{dim}"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["none", "cyclic", "block", "complete"]

    encoder.fit()

    return encoder

def getInlinedEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "inlined"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["0", "1"]
    encoder.fit()

    return encoder

def getResourceEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "resourceType"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["none", "bram_2P", "bram_1P"]
    encoder.fit()

    return encoder

def getNumCallsEncoder():
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "numCalls"
    encoder.method = EncoderMethod.LOG_NORMALIZED
    encoder.set_max(512)
    encoder.set_bias(2)
    return encoder

def getNumCallSitesEncoder():
    encoder = NormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "numCallSites"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(8)

    return encoder

def getDataTypeEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "datatype"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["int", "float", "NA", "void", "struct"]

    encoder.fit()

    return encoder

def getPipelinedEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "pipelined"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["0", "1"]

    encoder.fit()

    return encoder

def getPreviouslyPipelinedEncoder():
    encoder = OneHotEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "previouslyPipelined"
    encoder.method = EncoderMethod.ONE_HOT
    encoder.tags = ["0", "1"]

    encoder.fit()

    return encoder

def getBitwidthEncoder():
    encoder = NormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "bitwidth"
    encoder.method = EncoderMethod.NORMALIZED
    encoder.set_max(64)

    return encoder

def getTripcountEncoder():
    encoder = LogNormalizedEncoder()
    encoder.type = EncoderType.NODE
    encoder.label = "tripcount"
    encoder.method = EncoderMethod.LOG_NORMALIZED
    encoder.set_max(400*400)
    encoder.set_bias(10)

    return encoder



class Config():
    def __init__(self, graph_compiler_path=None):
        self.proxyPrograml = True

        self.encodeBBID = True
        self.encodeFuncID = True
        self.encodeEdgeOrder = True
        self.encodeNodeType = True

        self.absorbTypes = False
        self.absorbPragmas = False
        self.encodePreviousPipelined = False
        self.pipelinedUnroll = False

        self.convertAllocas = False

        self.ignoreControlFlow = False
        self.memoryOnlyControlFlow = False

        self.addNumCalls = False
        self.reduceIteratorBitwidths = False

        self.oneHotEncodeTypes = True
        self.indexFlowType = False

        self.encodeTripcount = False

        self.inline_on_graph = False


        if graph_compiler_path is None:
            self.invocation = "./../../../graph_compiler/bin/graph_compiler --hide_values"
        else:
            self.invocation = f"{graph_compiler_path} --hide_values"

    def save(self):
        self.encoders = []

        self.encoders.append(getDatasetIndexEncoder())
        self.encoders.append(getGraphTypeEncoder())

        if self.proxyPrograml:
            self.invocation += " --proxy_programl"

        if self.inline_on_graph:
            self.invocation += " --inline_functions"

        if self.encodeNodeType:
            self.invocation += " --add_node_type"
            self.encoders.append(getNodeTypeEncoder())


        #need for cfg
        self.invocation += " --add_bb_id"
        if self.encodeBBID:
            self.encoders.append(getBBEncoder())


        if self.encodeFuncID:
            self.encoders.append(getFuncIDEncoder())
            self.invocation += " --add_func_id"

        if self.encodeEdgeOrder:
            self.invocation += " --add_edge_order"
            self.encoders.append(getPositionEncoder())

        if self.convertAllocas:
            self.invocation += conversion_args
            self.keyTextTags = list(converted_nodes)
            self.flowTags = list(converted_edges)
        else:
            self.keyTextTags = list(baseline_nodes)
            self.flowTags =  list(baseline_edges)

        if self.oneHotEncodeTypes:
            if self.absorbTypes:
                self.encoders.append(getTypeEncoder(self.proxyPrograml, self.reduceIteratorBitwidths))
                self.invocation += " --absorb_types --one_hot_types"
            else:
                self.keyTextTags += list(type_nodes)
                if self.proxyPrograml:
                    self.keyTextTags += ["double"]
                else:
                    self.keyTextTags += ["f64"]
                self.invocation += " --one_hot_types"
        else:
            assert(self.absorbTypes)
            self.invocation += " --absorb_types"
            
            self.encoders.append(getDataTypeEncoder())
            self.encoders.append(getBitwidthEncoder())

            self.encoders.append(getTotalArrayWidthEncoder())

            for i in range(5):
                self.encoders.append(getArrayWidthEncoder(i))


        if self.absorbPragmas:
            self.invocation += " --absorb_pragmas"
            self.encoders.append(getPartitionEncoder(1))
            self.encoders.append(getPartitionEncoder(2))
            self.encoders.append(getInlinedEncoder())
            self.encoders.append(getPartitionFactorEncoder(1))
            self.encoders.append(getPartitionFactorEncoder(2))
            self.encoders.append(getPipelinedEncoder())
            self.encoders.append(getResourceEncoder())
            self.encoders.append(getUnrollFactorEncoder())
            for i in range(3):
                self.encoders.append(getHierarchicalUnrollFactorEncoder(i))

            self.encoders.append(getPreviouslyPipelinedEncoder())
            self.encoders.append(getTileEncoder())
            self.encoders.append(getPipelinedTypeEncoder())

            if self.pipelinedUnroll:
                self.invocation += " --add_unroll_from_pipeline"
        else:
            self.encoders.append(getNumericEncoder())
            self.keyTextTags +=  list(pragma_nodes)
            self.flowTags += list(["pragma"])

        if self.ignoreControlFlow:
            self.invocation += " --ignore_control_flow --ignore_call_edges"
        else:
            self.encoders.append(getFlowEncoder(self.flowTags))

        if self.memoryOnlyControlFlow:
            self.invocation += " --only_memory_control_flow"

        if self.addNumCalls:
            self.invocation += " --add_num_calls"
            self.encoders.append(getNumCallsEncoder())
            self.encoders.append(getNumCallSitesEncoder())

        if self.reduceIteratorBitwidths:
            self.invocation += " --reduce_iterator_bitwidth"

        if self.encodeTripcount:
            self.encoders.append(getTripcountEncoder())

        self.encoders.append(getKeyTextEncoder(self.keyTextTags))

       
class GraphConfigMayo(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()

class GraphConfigGalway(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()

class GraphConfigCavan(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()


class GraphConfigLouth(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.pipelinedUnroll = True

        self.absorbPragmas = True
        self.save()


# same as galway but without one hot encoded types
class GraphConfigKerry(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)

        self.oneHotEncodeTypes = False

        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()

class GraphConfigCork(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.oneHotEncodeTypes = False

        self.absorbTypes = True

        self.convertAllocas = True

        self.absorbPragmas = True

        self.pipelinedUnroll = True


        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()


class GraphConfigLimerick(Config):
    def __init__(self, graph_compiler_path=None):
        super().__init__(graph_compiler_path)
        self.oneHotEncodeTypes = False

        self.absorbTypes = True

        self.convertAllocas = True

        self.absorbPragmas = True

        self.encodeTripcount = True


        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.save()



