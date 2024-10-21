import mysql.connector
import pandas as pd
import json
import torch
import os

import balorgnn.generate.output_config as outConf

from functools import partial

class KernelData():
    def __init__(self, kernel_name, input_file):
        self.kernel_name = kernel_name

        self.input_file = input_file

        self.normalizer = {}
        self.denormalizer = {}

    def normalize(self, value, metric):
        assert(metric in self.normalizer)

        return self.normalizer[metric](value)
    
    def denormalize(self, value, metric):
        assert(metric in self.denormalizer)

        return self.denormalizer[metric](value)

    def get_pragmas(self, i):
        raise NotImplementedError()
    
    def get_values(self, metric, i):
        raise NotImplementedError()

    def get_num_values(self):
        raise NotImplementedError()
    

kernelMapDB4HLS = {   "gemm" : 297,
            "get_delta_matrix_weights3" : 365,
            "get_delta_matrix_weights1" : 366,
            "get_delta_matrix_weights2" : 367,
            "bfs" : 373,
            "update" : 382,
            "hist" : 383,
            "init" : 384,
            "sum_scan" : 385,
            "last_step_scan" : 386,
            "fft" : 389,
            "local_scan" : 390,
            "md_kernel" : 391,
            "twiddles8" : 396,
            "get_oracle_activations1" : 403,
            "get_oracle_activations2" : 404,
            "matrix_vector_product_with_bias_input_layer" : 405,
            "stencil3d" : 409,
            "ellpack" : 410,
            "bbgemm" : 412,
            "viterbi" : 415,
            "aes_shiftRows" : 424,
            "ms_mergesort" : 436,
            "merge" : 437,
            "add_bias_to_activations" : 440,
            "aes256_encrypt_ecb" : 441,
            "aes_expandEncKey" : 442,
            "ss_sort" : 443,
            "stencil" : 444,
            "soft_max" : 446,
            "take_difference" : 447,
            "matrix_vector_product_with_bias_output_layer" : 449,
            "update_weights" : 450,
            "backprop" : 451,
            "aes_addRoundKey" : 455,
            "aes_addRoundKey_cpy" : 457,
            "aes_mixColumns" : 459,
            "aes_subBytes" : 460,
            "matrix_vector_product_with_bias_second_layer" : 461,
        }    

class KernelDataDB4HLS(KernelData):
    def __init__(self, kernel_name, base_path):
        self.metrics = ["LUTs", "FFs", "Latency", "DSPs", "BRAMs", "Clock"]

        if base_path.endswith('/'):
            base_path = base_path[:-1]

        input_file = f"{base_path}/{kernel_name}.cpp"

        super().__init__(kernel_name, input_file)

        kernel_id = kernelMapDB4HLS[kernel_name]

        cnx = mysql.connector.connect(user='user', password='password', host='localhost', auth_plugin='mysql_native_password')
        cursor = cnx.cursor()

        query = f"""
            SELECT 
            db4hls.configuration.config_script, 
            db4hls.resource_results.hls_lut,
            db4hls.resource_results.hls_ff,
            db4hls.performance_results.average_latency,
            db4hls.resource_results.hls_bram,
            db4hls.resource_results.hls_dsp,
            db4hls.performance_results.estimated_clock
            FROM db4hls.configuration
            JOIN db4hls.configuration_space ON db4hls.configuration.id_configuration_space = db4hls.configuration_space.id_configuration_space
            JOIN db4hls.implementation ON db4hls.configuration.hash_configuration = db4hls.implementation.hash_configuration
            LEFT JOIN db4hls.resource_results ON db4hls.implementation.id_resource_results = db4hls.resource_results.id_resource_result
            LEFT JOIN db4hls.performance_results ON db4hls.implementation.id_performance_results = db4hls.performance_results.id_performance_result
            WHERE db4hls.configuration_space.id_configuration_space = {kernel_id} AND db4hls.performance_results.average_latency > 0
            """

        cursor.execute(query)
        
        self.data = cursor.fetchall()
        self.data = pd.DataFrame(self.data)
        self.data.columns = ["pragmas", "LUTs", "FFs", "Latency", "BRAMs", "DSPs", "Clock"]

    def get_pragmas(self, i):
        return self.data["pragmas"].iloc[i]
    
    def get_values(self, metric, i):
        if metric in self.metrics:
            return self.data[metric].iloc[i]
        
        assert(False, "Metric not recognized for DB4HLS")


    def get_num_values(self):
        return len(self.data)
    

def extract_config(target_name, config_file):
    found_config = False
    config_lines = []
    full = []
    with open(config_file, 'r') as f:
        for line in f:
            if target_name in line:
                # print(target_name, line)
                found_config = True
            if found_config:
                full.append(line)
                if line.strip().startswith("set_directive_resource"):
                    config_lines.append(line.strip())
                elif line.strip().startswith("set_directive_array_partition"):
                    config_lines.append(line.strip())
                elif line.strip().startswith("set_directive_interface"):
                    config_lines.append(line.strip())
                elif line.strip().startswith("set_directive_pipeline"):
                    config_lines.append(line.strip())
                elif line.strip().startswith("set_directive_unroll"):
                    config_lines.append(line.strip())
                elif line.strip().startswith("csynth_design"):
                    config_lines.append(line.strip())
                    break  # Assuming the configuration ends here
    out = '\n'.join(config_lines)
    return out

class KernelDataPolybench(KernelData):
    def __init__(self, kernel_name, base_path):
        self.metrics = ["Total LUTs", "Logic LUTs", "SRLs", "FFs", "RAMB36", "RAMB18", "DSP48 Blocks", "Latency", "Dynamic Power"]

        if base_path.endswith('/'):
            base_path = base_path[:-1]

        input_file = f"{base_path}/{kernel_name}/{kernel_name}.c"

        super().__init__(kernel_name, input_file)


        self.data = pd.read_csv(f"{base_path}/{kernel_name}/post_implementation_info_latency.csv", sep='\t')
        get_pragma = partial(extract_config, config_file=f"{base_path}/{kernel_name}/hls.tcl")
        self.data['pragmas'] = self.data['prj'].apply(get_pragma)

    def get_pragmas(self, i):
        return self.data["pragmas"].iloc[i]
    
    def get_values(self, metric, i):
        if metric == "Latency":
            return self.data["latency"].iloc[i]
        if metric == "Dynamic Power":
            return self.data["vivado_dynamic_pwr(mW)"].iloc[i]
        
        if metric not in self.metrics:
            assert(False, "Metric not recognized for Polybench")

        return self.data[metric].iloc[i]


    def get_num_values(self):
        return len(self.data)
    
class KernelDataPowerGear(KernelData):
    def __init__(self, kernel_name, base_path):
        self.metrics = ["Total Power", "Static Power", "Dynamic Power"]

        if base_path.endswith('/'):
            base_path = base_path[:-1]

        input_file = f"{base_path}/{kernel_name}/{kernel_name}.c"

        super().__init__(kernel_name, input_file)


        self.data = pd.read_csv(f"{base_path}/{kernel_name}/power_measurement.csv")
        get_pragma = partial(extract_config, config_file=f"{base_path}/{kernel_name}/script_0.tcl")
        self.data['pragmas'] = self.data['prj'].apply(get_pragma)

    def get_pragmas(self, i):
        return self.data["pragmas"].iloc[i]
    
    def get_values(self, metric, i):
        if metric == "Total Power":
            return self.data["total_pwr(uW)"].iloc[i]
        if metric == "Static Power":
            return self.data["static_pwr(uW)"].iloc[i]
        if metric == "Dynamic Power":
            return self.data["dynamic_pwr(mW)"].iloc[i]
        
        if metric not in self.metrics:
            assert(False, "Metric not recognized for Polybench")

        return self.data[metric].iloc[i]


    def get_num_values(self):
        return len(self.data)
    


class KernelDataVast(KernelData):
    def __init__(self, kernel_name, base_path, output_config_name):
        if base_path.endswith('/'):
            base_path = base_path[:-1]

        input_file = f"{base_path}/sources/{kernel_name}.c"

        # function names in the file start with kernel_
        # to avoid starting functions with numbers
        super().__init__(f"kernel_{kernel_name}", input_file)

        if(output_config_name == outConf.OutputConfigNames.VAST_18):
            self.folder = "v18"
        elif (output_config_name == outConf.OutputConfigNames.VAST_20):
            self.folder = "v20"
        elif (output_config_name == outConf.OutputConfigNames.VAST_21):
            self.folder = "v21"
        else:
            print("Unrecognized vast output config")
            quit()
        with open(f"{base_path}/designs/{self.folder}/{kernel_name}.json", 'r') as file:
            data = json.load(file)

        pragmas = [data[key]['point'] for key in data]
        util = [data[key]['res_util'] for key in data]
        util_lut = [row["util-LUT"] for row in util]
        util_ff = [row["util-FF"] for row in util]
        util_dsp = [row["util-DSP"] for row in util]
        util_bram = [row["util-BRAM"] for row in util]
        latency = [data[key]["perf"] for key in data]
        valid = [data[key]["valid"] for key in data]

        self.data = pd.DataFrame({
                    "pragmas" : pragmas,
                    })
        
        column_types = [" valid", " all"]

        join_types = ["", " join"]

        for column_type in column_types:
            for join_type in join_types:
                self.data[f"LUTs{join_type}{column_type}"] = util_lut
                self.data[f"FFs{join_type}{column_type}"] = util_ff
                self.data[f"BRAMs{join_type}{column_type}"] = util_bram
                self.data[f"DSPs{join_type}{column_type}"] = util_dsp
                self.data[f"Latency{join_type}{column_type}"] = latency
        
        self.data["Valid"] = valid
        self.data["Valid join"] = valid

        synthesized = self.data[f"Latency all"] > 0
        self.data["Synthesized"] = synthesized
        self.data["Synthesized join"] = synthesized

        oversized = (self.data["Latency all"] > 0) & (~self.data["Valid"])
        self.data["Oversized"] = oversized
        self.data["Oversized join"] = oversized

        unconditional_columns = ['Valid', 'Valid join', "Synthesized", "Synthesized join", "Oversized", "Oversized join", "pragmas"]

        self.data.loc[self.data['Valid'] == False, self.data.columns.difference(unconditional_columns)] = 0
        
    def get_pragmas(self, i):
        return self.data["pragmas"].iloc[i]
    
    def get_values(self, metric, i):
        if metric not in self.data.columns:
            assert(False, "Metric not recognized for VAST")
        return self.data[metric].iloc[i]

    def get_num_values(self):
        return len(self.data)
    
class KernelDataVastCustom(KernelDataVast):
    def __init__(self, kernel_name, base_path, output_config_name):
        self.raw_kernel_name = kernel_name
        self.use_output_graphs = True
        super().__init__(kernel_name, base_path, output_config_name)
        self.data['original_index'] = self.data.index    
        self.data = self.data[[os.path.exists(f"../../vast/translate/output_sources_{self.folder}/{self.raw_kernel_name}/{i}.cpp") for i in range(len(self.data))]]

    def get_pragmas(self, i):
        if self.use_output_graphs:
            index = self.data["original_index"].iloc[i]
            self.input_file = f"../../vast/translate/output_sources_{self.folder}/{self.raw_kernel_name}/{index}.cpp"
        else:
            self.input_file = f"../../vast/sources/{self.raw_kernel_name}.c"

        return self.data["pragmas"].iloc[i]
    
     
    # def get_use_in_loss(self, i):
    #     values = []
    #     for metric in self.metrics:
    #         if metric in ["LUTs", "FFs", "DSPs", "BRAMs"]:
    #             value = self.get_values(metric, i)
    #             values.append(value != 0)
    #         else:
    #             values.append(True)
    #     return torch.tensor(values)
     

class KernelDataGNNDSE(KernelData):
    def __init__(self, kernel_name, base_path):
        self.metrics = [ "LUTs", "FFs", "DSPs", "BRAMs", "Latency"]

        if base_path.endswith('/'):
            base_path = base_path[:-1]

        input_file = f"{base_path}/sources/{kernel_name}.c"

        # function names in the file start with kernel_
        # to avoid starting functions with numbers
        super().__init__(f"kernel_{kernel_name}", input_file)

        with open(f"{base_path}/designs/v18/{kernel_name}.json", 'r') as file:
            data = json.load(file)

  
        pragmas = [data[key]['point'] for key in data]
        util = [data[key]['res_util'] for key in data]
        util_lut = [row["util-LUT"] for row in util]
        util_ff = [row["util-FF"] for row in util]
        util_dsp = [row["util-DSP"] for row in util]
        util_bram = [row["util-BRAM"] for row in util]
        valid = [data[key]["valid"] for key in data]
        latency = [data[key]["perf"] for key in data]

        self.data = pd.DataFrame({
                           "pragmas" : pragmas,
                           "LUTs": util_lut, 
                           "FFs": util_ff,
                           "DSPs": util_dsp,
                           "BRAMs" : util_bram,
                           "Valid" : valid,
                           "Latency" : latency
                           })
        

        self.data = self.data[self.data["Valid"]]
        self.data = self.data[self.data["LUTs"] > 0]

    def get_pragmas(self, i):
        return self.data["pragmas"].iloc[i]
    
    def get_values(self, metric, i):

        if metric not in self.metrics:
            assert(False, "Metric not recognized for GNN-DSE")

        return self.data[metric].iloc[i]

    def get_num_values(self):
        return len(self.data)
    
