import argparse
import time
import multiprocessing
import os

import subprocess
import pygraphviz as pgv

import torch.nn.functional as F

from enum import Enum

import shutil

from balorgnn.generate.apply_directives import apply_vitis_directives, apply_merlin_directives
from balorgnn.generate.kernel_data import KernelDataDB4HLS, KernelDataPolybench, KernelDataPowerGear, KernelDataVast, KernelDataGNNDSE, KernelDataVastCustom
import balorgnn.generate.graph_to_data as graphToData
import balorgnn.generate.graph_config as graphConf
import balorgnn.generate.output_config as outputConf

from balorgnn.data.dataset import CustomData

from functools import partial

from collections import defaultdict

import torch

from balorgnn.generate.output_config import OutputConfigNames

class KernelList(Enum):
    RED = 0
    ORANGE = 1
    PINK = 2
    INDIGO = 3
    VIOLET = 4
    RUBY = 5
    SAPPHIRE = 6
    GOLD = 7
    SILVER = 8
    PLATINUM = 9
    CRYSTAL = 10

class GraphConfigNames(Enum):
    MAYO = 0
    GALWAY = 1
    CAVAN = 2
    LOUTH = 3
    KERRY = 4
    CORK = 5
    LIMERICK = 6


class DatasetGenerator():
    def __init__(self, dataset_folder, graph_compiler, inputs_folder, output_folder, graph_config_name, kernel_list, combine_vast, all_vast_21, merlin_only, valid_only, no_regen):
        self.num_processes = 16
        
        self.inputs_folder = inputs_folder

        self.combine_vast = combine_vast

        self.all_vast_21 = all_vast_21

        self.merlin_only = merlin_only
        self.valid_only = valid_only

        self.no_regen = no_regen
        

        self.temp_dir = "tmp"
        os.makedirs(self.temp_dir, exist_ok=True)

        if graph_config_name == GraphConfigNames.MAYO:
            graph_config = graphConf.GraphConfigMayo(graph_compiler)
            graph_config_folder = "mayo"
        elif graph_config_name == GraphConfigNames.CAVAN:
            graph_config = graphConf.GraphConfigCavan(graph_compiler)
            graph_config_folder = "cavan"
        elif graph_config_name == GraphConfigNames.GALWAY:
            graph_config = graphConf.GraphConfigGalway(graph_compiler)
            graph_config_folder = "galway"
        elif graph_config_name == GraphConfigNames.LOUTH:
            graph_config = graphConf.GraphConfigLouth(graph_compiler)
            graph_config_folder = "louth"
        elif graph_config_name == GraphConfigNames.KERRY:
            graph_config = graphConf.GraphConfigKerry(graph_compiler)
            graph_config_folder = "kerry"
        elif graph_config_name == GraphConfigNames.CORK:
            graph_config = graphConf.GraphConfigCork(graph_compiler)
            graph_config_folder = "cork"
        elif graph_config_name == GraphConfigNames.LIMERICK:
            graph_config = graphConf.GraphConfigLimerick(graph_compiler)
            graph_config_folder = "limerick"

        self.invocation = graph_config.invocation
        self.graph_encoders = graph_config.encoders

        self.kernels = defaultdict(list)
        self.outputs = defaultdict(list)


        if KernelList.RED in kernel_list:
            red_kernel_list = ["gemm","bfs","update","hist","init","sum_scan","last_step_scan","fft","local_scan","md_kernel","twiddles8","get_oracle_activations1","get_oracle_activations2","matrix_vector_product_with_bias_input_layer","stencil3d","ellpack","bbgemm","viterbi","aes_shiftRows","ms_mergesort","merge","add_bias_to_activations","aes256_encrypt_ecb","aes_expandEncKey","ss_sort","stencil","soft_max","take_difference","matrix_vector_product_with_bias_output_layer","update_weights","backprop","aes_addRoundKey","aes_addRoundKey_cpy","aes_mixColumns","aes_subBytes","matrix_vector_product_with_bias_second_layer"]

            self.kernels[OutputConfigNames.DB4HLS].extend(red_kernel_list)
            self.outputs[OutputConfigNames.DB4HLS] = outputConf.OutputConfigDB4HLS().metrics
        if KernelList.ORANGE in kernel_list:
            orange_kernel_list = ["gemm", "bicg", "gesummv", "k2mm", "k3mm", "syr2k", "syrk"]
            self.kernels[OutputConfigNames.ML4ACCEL].extend(orange_kernel_list)
            self.outputs[OutputConfigNames.ML4ACCEL] = outputConf.OutputConfigML4ACCEL().metrics
        if KernelList.PINK in kernel_list :
            pink_kernel_list = ["atax", "gemm", "bicg", "gesummv", "k2mm", "k3mm", "mvt", "syr2k", "syrk"]
            self.kernels[OutputConfigNames.POWERGEAR].extend(pink_kernel_list)
            self.outputs[OutputConfigNames.POWERGEAR] = outputConf.OutputConfigPowergear().metrics

        if KernelList.VIOLET in kernel_list:

            violet_kernel_list = ["3mm", "2mm", "adi", "aes", "atax", "bicg", "correlation", "doitgen", "fdtd_2d", "gemm_blocked", "gemver", "gesummv", "heat_3d", "jacobi_1d", "jacobi_2d", "md", "mvt", "nw", "seidel_2d","spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "syr2k", "syrk", "trmm"]
            self.kernels[OutputConfigNames.GNNDSE].extend(violet_kernel_list)
            self.outputs[OutputConfigNames.GNNDSE] = outputConf.OutputConfigGNNDSE().metrics

        if KernelList.INDIGO in kernel_list:
            indigo_kernel_list = ["3mm", "2mm", "adi", "aes", "atax", "atax_medium", "bicg", "bicg_medium", "correlation",  "doitgen", "doitgen_red", "fdtd_2d", "gemm_blocked", "gemm_ncubed", "gemm_p", "gemver", "gemver_medium", "gesummv", "gesummv_medium", "heat_3d", "jacobi_1d", "jacobi_2d", "md", "mvt", "mvt_medium", "nw", "seidel_2d", "spmv_crs", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "symm_opt_medium", "syr2k", "syrk", "trmm", "trmm_opt"]

            self.kernels[OutputConfigNames.VAST_18].extend(indigo_kernel_list)
            self.outputs[OutputConfigNames.VAST_18] = outputConf.OutputConfigVast().metrics

        if KernelList.RUBY in kernel_list:
            ruby_kernel_list = ["2mm", "aes", "atax", "bicg", "bicg_large", "correlation", "covariance", "doitgen", "doitgen_red", "fdtd_2d_large", "gemm_blocked", "gemm_ncubed", "gemm_p", "gemm_p_large", "gemver", "gesummv", "mvt", "nw", "spmv_crs", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "symm_opt", "syr2k", "syrk", "trmm", "trmm_opt"]
            

            self.kernels[OutputConfigNames.VAST_20].extend(ruby_kernel_list)
            self.outputs[OutputConfigNames.VAST_20] = outputConf.OutputConfigVast().metrics

        if KernelList.SAPPHIRE in kernel_list:
            sapphire_kernel_list = ["3mm", "2mm", "adi", "aes", "atax", "atax_medium", "bicg", "bicg_large", "correlation", "covariance", "doitgen", "doitgen_red", "fdtd_2d", "fdtd_2d_large", "gemm_blocked", "gemm_ncubed", "gemm_p","gemm_p_large", "gemver", "gemver_medium", "gesummv", "gesummv_medium", "heat_3d", "jacobi_1d", "jacobi_2d", "mvt", "mvt_medium", "nw", "seidel_2d", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "symm_opt", "symm_opt_medium", "syr2k", "syrk", "trmm", "trmm_opt"]
            
            self.kernels[OutputConfigNames.VAST_21].extend(sapphire_kernel_list)
            self.outputs[OutputConfigNames.VAST_21] = outputConf.OutputConfigVast().metrics
        

        if KernelList.GOLD in kernel_list:
            gold_kernel_list = ["2mm", "3mm", "adi", "aes", "atax", "atax_medium", "bicg_large", "bicg", "bicg_medium", "correlation", "covariance", "doitgen_red", "doitgen", "fdtd_2d_large", "fdtd_2d", "gemm_blocked", "gemm_ncubed", "gemm_p_large", "gemm_p", 'gemver_medium', "gemver", "gesummv_medium", "gesummv", "heat_3d", "jacobi_1d", "jacobi_2d", "mvt_medium", "mvt", "nw", "seidel_2d", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm_opt_medium", "symm_opt", "symm", "syr2k", "syrk", "trmm_opt", "trmm"]

            self.kernels[OutputConfigNames.VAST_CUSTOM_21].extend(gold_kernel_list)
            # self.outputs[OutputConfigNames.VAST_21] = outputConf.OutputConfigVast().metrics

        if KernelList.SILVER in kernel_list:
            silver_kernel_list = ["2mm", "aes", "atax", "bicg", "bicg_large", "correlation", "covariance", "doitgen", "doitgen_red", "fdtd_2d_large", "gemm_blocked", "gemm_ncubed", "gemm_p", "gemm_p_large", "gemver", "gesummv", "mvt", "nw", "spmv_crs", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "symm_opt", "syr2k", "syrk", "trmm", "trmm_opt"]

            self.kernels[OutputConfigNames.VAST_CUSTOM_20].extend(silver_kernel_list)
            # self.outputs[OutputConfigNames.VAST_20] = outputConf.OutputConfigVast().metrics

        if KernelList.PLATINUM in kernel_list:
            platinum_kernel_list = ["2mm", "3mm", "adi", "aes", "atax", "atax_medium", "bicg", "bicg_medium", "correlation",  "doitgen", "doitgen_red", "fdtd_2d", "gemm_blocked", "gemm_ncubed", "gemm_p", "gemver", "gemver_medium", "gesummv", "gesummv_medium", "heat_3d", "jacobi_1d", "jacobi_2d", "md", "mvt", "mvt_medium", "nw", "seidel_2d", "spmv_crs", "spmv_ellpack", "stencil_2d", "stencil_3d", "symm", "symm_opt_medium", "syr2k", "syrk", "trmm", "trmm_opt"]

            self.kernels[OutputConfigNames.VAST_CUSTOM_18].extend(platinum_kernel_list)
            # self.outputs[OutputConfigNames.VAST_18] = outputConf.OutputConfigVast().metrics
            # self.outputs[OutputConfigNames.VAST_18] = outputConf.OutputConfigVast().metrics

        if KernelList.CRYSTAL in kernel_list:
            crystal_kernel_list = ["3mm", "atax_medium", "covariance", "fdtd_2d", "gemm_p", "gemver_medium", "jacobi_2d", "symm_opt", "trmm_opt", "syr2k"]

            # self.kernels[OutputConfigNames.VAST_CUSTOM_18].extend(crystal_kernel_list)
            # self.kernels[OutputConfigNames.VAST_CUSTOM_20].extend(crystal_kernel_list)
            self.kernels[OutputConfigNames.VAST_CUSTOM_21].extend(crystal_kernel_list)

            # self.outputs[OutputConfigNames.VAST_18] = outputConf.OutputConfigVast().metrics



        if combine_vast:
            self.outputs[OutputConfigNames.VAST_ALL] = outputConf.OutputConfigVast(join=True).metrics



        self.all_outputs = []
        self.output_shift = {}

        # store the shift and length of the concatenated outputs vector
        # so that the right number of MLPs is generated
        # and so the right metric can be extracted after inference
        for key in self.outputs:
            self.output_shift[key] = len(self.all_outputs)
            self.all_outputs.extend(self.outputs[key])

        self.output_length = len(self.all_outputs)


        self.output_dir = f"{dataset_folder}/{output_folder}/{graph_config_folder}/"
        os.makedirs(self.output_dir, exist_ok=True)

    def generateData(self):
        base_data_id = 0
        start_time = time.time()

        for output_config_name in self.kernels:
            self.set_output_config(output_config_name)
            for kernel_num, kernel in enumerate(self.kernels[output_config_name]):


                print(f"Processing kernel: {kernel}, {kernel_num+1}/{len(self.kernels[output_config_name])}")
                
                kernel_data = self.kernel_data(kernel)

                if(len(kernel_data.data) == 0):
                    continue

                if self.valid_only and self.is_vast_config:
                    kernel_data.data = kernel_data.data[kernel_data.data["Valid"]]

                a = time.time()

                with multiprocessing.Pool(processes=self.num_processes) as pool:
                    input = [(kernel_data, base_data_id, i) for i in range(self.num_processes)]
                    out = pool.starmap_async(self.run_cpu_thread, input, error_callback=DatasetGenerator.custom_error_callback)

                    # Close the pool to prevent any more tasks from being submitted
                    pool.close()
                    out.wait()
                    # Join the worker processes to clean up resources
                    pool.join()

                base_data_id = base_data_id + kernel_data.get_num_values()
                b = time.time()

                print(f"Completed in {b-a} seconds")
        end_time = time.time()
        print(f"Total time: {end_time-start_time}")
        print(f"Total Designs: {base_data_id}")

        # shutil.rmtree(self.temp_dir)

    def custom_error_callback(error):
        print(f'Got error: {error}')


    def set_output_config(self, output_config_name):
        self.output_config_name = output_config_name
        self.is_vast_config = False
        if output_config_name == OutputConfigNames.DB4HLS:
            self.output_config = outputConf.OutputConfigDB4HLS()
            self.kernel_data = partial(KernelDataDB4HLS, base_path=f"{self.inputs_folder}/machsuite")
            self.apply_directives = apply_vitis_directives
        if output_config_name == OutputConfigNames.ML4ACCEL:
            self.output_config = outputConf.OutputConfigML4ACCEL()
            self.kernel_data = partial(KernelDataPolybench, base_path=f"{self.inputs_folder}/polybench")
            self.apply_directives = apply_vitis_directives
        if output_config_name == OutputConfigNames.POWERGEAR:
            self.output_config = outputConf.OutputConfigPowergear()
            self.kernel_data = partial(KernelDataPowerGear, base_path=f"{ self.inputs_folder}/powergear")
            self.apply_directives = apply_vitis_directives

        is_vast = output_config_name == OutputConfigNames.VAST_18
        is_vast = is_vast or output_config_name == OutputConfigNames.VAST_20
        is_vast = is_vast or output_config_name == OutputConfigNames.VAST_21
        if is_vast:
            self.is_vast_config = True
            self.output_config = outputConf.OutputConfigVast()
            self.kernel_data = partial(KernelDataVast, base_path=f"{self.inputs_folder}/vast", output_config_name=output_config_name)
            self.apply_directives = apply_merlin_directives

        if output_config_name == OutputConfigNames.VAST_CUSTOM_18:
            self.output_config = outputConf.OutputConfigVast(join=True)
            self.kernel_data = partial(KernelDataVastCustom, base_path=f"{self.inputs_folder}/vast", output_config_name=OutputConfigNames.VAST_18)
            self.apply_directives = apply_merlin_directives
            self.output_config_name = OutputConfigNames.VAST_18
            self.is_vast_config = True

        if output_config_name == OutputConfigNames.VAST_CUSTOM_20:
            self.output_config = outputConf.OutputConfigVast(join=True)
            self.kernel_data = partial(KernelDataVastCustom, base_path=f"{self.inputs_folder}/vast", output_config_name=OutputConfigNames.VAST_20)
            self.apply_directives = apply_merlin_directives
            self.output_config_name = OutputConfigNames.VAST_20
            self.is_vast_config = True

        if output_config_name == OutputConfigNames.VAST_CUSTOM_21:
            self.output_config = outputConf.OutputConfigVast(join=True)
            self.kernel_data = partial(KernelDataVastCustom, base_path=f"{self.inputs_folder}/vast", output_config_name=OutputConfigNames.VAST_21)
            self.apply_directives = apply_merlin_directives
            self.output_config_name = OutputConfigNames.VAST_21     
            self.is_vast_config = True  


        if output_config_name == OutputConfigNames.GNNDSE:
            self.output_config = outputConf.OutputConfigGNNDSE()
            self.kernel_data = partial(KernelDataGNNDSE, base_path=f"{self.inputs_folder}/vast")
            self.apply_directives = apply_merlin_directives

    def run_cpu_thread(self, kernel_data, base_data_id, thread_id):
        try:
            # each thread processes integer multiples of the the thread ID
            for i in range(thread_id, kernel_data.get_num_values(), self.num_processes):
                
                if self.no_regen:
                    if os.path.exists(f"{self.output_dir}/data_{base_data_id + i}.pt"):
                        continue

                # generate .cpp file of kernel with pragmas
                pragmadFile = f"{self.temp_dir}/{thread_id}.cpp"

                kernel_data.use_output_graphs = True
                self.apply_directives(kernel_data, i, pragmadFile)

                mask = self.get_mask(self.output_config_name)
                use_in_loss_mask = self.output_config.get_use_in_loss(i, kernel_data)

                # run graph compiler on cpp file
                output_config_value = self.get_output_config_value(self.output_config_name)

                full_invocation = self.invocation + f" --top {kernel_data.kernel_name} --src {pragmadFile} --datasetIndex {output_config_value} --graphType 0"

                
                graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)

                # parse graph compiler graph output to pgv
                graph = pgv.AGraph(string=graphOutput.stdout)

                # process pgv graph representation to pytorch geometric representation 
                node_array, edge_index, edge_attr = graphToData.make_graph_arrays(self.graph_encoders, graph)

                # since some methods allow pragmas to add nodes to the graph, the list of which nodes belong to which BB
                # must be made per graph
                bb_id_list = graphToData.make_bb_id_list(graph)

                cfg = graphToData.make_cfg_from_graph(graph)
                cfg1 = cfg

                graph1 = graph
                graph2 = None

                #####################
                #  Duplication
                #####################


                if (not self.merlin_only) and self.is_vast_config:

                    kernel_data.use_output_graphs = False
                    self.apply_directives(kernel_data, i, pragmadFile)

                    output_config_value = self.get_output_config_value(self.output_config_name)

                    # run graph compiler on cpp file                
                    full_invocation = self.invocation + f" --top {kernel_data.kernel_name} --src {pragmadFile} --datasetIndex {output_config_value} --graphType 1"

                    graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)

                    # parse graph compiler graph output to pgv
                    graph = pgv.AGraph(string=graphOutput.stdout)
                    graph2 = graph

                    # process pgv graph representation to pytorch geometric representation 
                    node_array_small, edge_index_small, edge_attr_small = graphToData.make_graph_arrays(self.graph_encoders, graph)

                    # since some methods allow pragmas to add nodes to the graph, the list of which nodes belong to which BB
                    # must be made per graph
                    bb_id_list_small = graphToData.make_bb_id_list(graph)

                    cfg2 = graphToData.make_cfg_from_graph(graph)

                    ###########################
                    # Combine
                    ###########################

                    node_array = torch.cat((node_array, node_array_small), dim=0)
                    edge_index = torch.cat((edge_index, edge_index_small), dim=1)
                    edge_attr = torch.cat((edge_attr, edge_attr_small), dim=0)

                    cfg2.cfg_edge_index = cfg2.cfg_edge_index + cfg.num_bbs

                    bb_id_list_small = bb_id_list_small + cfg.num_bbs
                    bb_id_list = torch.cat((bb_id_list, bb_id_list_small), dim=0)

                    num_bbs = cfg.num_bbs + cfg2.num_bbs

                    cfg_edge_index = torch.cat((cfg.cfg_edge_index, cfg2.cfg_edge_index), dim=1)
                    bb_batch = torch.cat((cfg.bb_batch, cfg2.bb_batch), dim=0)

                    cfg = graphToData.CFG(cfg_edge_index, num_bbs, bb_batch)


                ############################


                output_array = self.output_config.process_output_vector(kernel_data, i)

                # when combining vast we write the output array twice to the output vector
                # once to the version-specialized MLP and once to the all-vast MLP
                # if self.combine_vast and self.is_vast_config:
                #     output_array = torch.concat([output_array, output_array], dim=1)
                
                
                real_output_array = torch.zeros(self.output_length)
                real_output_array[mask[0]] = output_array
                real_output_array = real_output_array.unsqueeze(0)

                # when combining vast we write the output array twice to the output vector
                # once to the version-specialized MLP and once to the all-vast MLP
                # if self.combine_vast and self.is_vast_config:
                #     use_in_loss_mask = torch.concat([use_in_loss_mask, use_in_loss_mask])

                real_use_in_loss_mask = torch.zeros(self.output_length, dtype=torch.bool)
                real_use_in_loss_mask[mask[0]] = use_in_loss_mask
                real_use_in_loss_mask = real_use_in_loss_mask.unsqueeze(0)


                # if not combine vast should use the actual config to get the shift
                # shift = torch.tensor(self.output_shift[self.output_config_name])

                if self.combine_vast and self.is_vast_config:
                    shift = torch.tensor(self.output_shift[outputConf.OutputConfigNames.VAST_ALL])
                else:
                    shift = torch.tensor(self.output_shift[self.output_config_name])

                if self.combine_vast:
                    join_shift = torch.tensor(self.output_shift[outputConf.OutputConfigNames.VAST_ALL])
                else:
                    join_shift = torch.tensor(0)

                string = f"node_array: {node_array.shape} \n"
                string = string + f"edge_index: {edge_index.shape} \n"
                string = string + f"real_output_array: {real_output_array.shape} \n"
                string = string + f"bb_id_list: {bb_id_list.shape} \n"
                string = string + f"bb_batch: {cfg.bb_batch.shape} \n"

                # print(string)

                from torch_scatter import scatter_add
                cfg_node = scatter_add(node_array, bb_id_list, dim=0)
                graph = scatter_add(cfg_node, cfg.bb_batch, dim=0)
                string = f"cfg_node: {cfg_node.shape} \n"
                string = string + f"cfg_node: {graph.shape} \n"


                string = string + "\n"


                # print(string)
                # quit()

                output_config_name = torch.tensor(self.output_config_name.value)


                data = CustomData(
                            x=node_array, 
                            edge_index=edge_index, 
                            edge_attr=edge_attr, 
                            y=real_output_array, 
                            cfg_edge_index= cfg.cfg_edge_index,
                            bb_id_list = bb_id_list,
                            num_bbs = cfg.num_bbs,
                            bb_batch = cfg.bb_batch,
                            output_config = self.output_config,
                            kernel = kernel_data.kernel_name,
                            pragmas = str(kernel_data.get_pragmas(i)),
                            output_config_name = output_config_name,
                            all_outputs = self.all_outputs,
                            shift = shift,
                            join_shift = join_shift,
                            use_in_loss_mask = real_use_in_loss_mask,
                            # graph1 = str(graph1),
                            # graph2 = str(graph2)
                            )


                
                            
                torch.save(data, f"{self.output_dir}/data_{base_data_id + i}.pt")
                # print(base_data_id + i)
        except Exception as e:
            modified_exception = ValueError(f"There was an error in {kernel_data.kernel_name} {i}: {e}")
            raise modified_exception from e 

    def get_mask(self, output_config):
        mask = []
        for key in self.outputs:
            if key != outputConf.OutputConfigNames.VAST_ALL:
                value = key == output_config
                for _ in self.outputs[key]:
                    mask.append(value)
            else:
                for _ in self.outputs[key]:
                    mask.append(self.is_vast_config)
        mask = torch.tensor([mask], dtype=torch.bool)
        return mask

    def get_output_config_value(self, output_config_name):
        if not self.all_vast_21:
            return output_config_name.value
        else:
            return OutputConfigNames.VAST_21.value

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of graphs for GNN QoR Estimation")

    parser.add_argument('--dataset_folder', help='Folder to store dataset in.')
    parser.add_argument('--graph_compiler', help='Location of graph compiler binary')
    parser.add_argument('--inputs_folder', help='Location of input folder holding c++ files')



    parser.add_argument('--graph_config_mayo', action='store_true', help='Run with graph config mayo.')
    parser.add_argument('--graph_config_galway', action='store_true', help='Run with graph config galway (reduced graph, absorbed pragmas).')
    parser.add_argument('--graph_config_cavan', action='store_true', help='Run with graph config cavan (reduced graph but with pragma nodes).')
    parser.add_argument('--graph_config_louth', action='store_true', help='Run with graph config louth (reduced graph, pragmas absorbed, pipelined unroll).')
    parser.add_argument('--graph_config_kerry', action='store_true', help='Run with graph config kerry (same as galway but without one-hot-encoding types).')
    parser.add_argument('--graph_config_cork', action='store_true', help='Run with graph config cork (kerry + pipelined unroll.)')
    parser.add_argument('--graph_config_limerick', action='store_true', help='Run with graph config limerick (kerry + tripcount.)')


    parser.add_argument('--kernels_red', action='store_true', help='Run with kernel group red (all machsuite kernels).')
    parser.add_argument("--kernels_orange", action='store_true', help='Run with kernel group orange (all from ml4accel polybench).')
    parser.add_argument("--kernels_pink", action='store_true', help='Run with kernel group pink (all from powergear polybench).')


    parser.add_argument("--kernels_violet", action='store_true', help='Run with kernel group violet (all from VAST 18, with GNNDSE scaling).')
    parser.add_argument("--kernels_indigo", action='store_true', help='Run with kernel group indigo (all from VAST 18).')
    parser.add_argument("--kernels_ruby", action='store_true', help='Run with kernel group ruby (all from VAST 20).')
    parser.add_argument("--kernels_sapphire", action='store_true', help='Run with kernel group sapphire (all from VAST 21).')


    parser.add_argument("--kernels_gold", action='store_true', help='Run with kernel group gold (all from v21 with input graphs)).')
    parser.add_argument("--kernels_silver", action='store_true', help='Run with kernel group silver (all from v20 with input graphs)).')
    parser.add_argument("--kernels_platinum", action='store_true', help='Run with kernel group platinum (all from v18 with input graphs)).')

    parser.add_argument("--kernels_crystal", action='store_true', help='To optimize for stage 2 from v18 with input graphs.')




    parser.add_argument("--combine_vast", action='store_true', help='Generate outputs that are trained on all VAST datasets, regardless of version')
    parser.add_argument("--all_vast_21", action='store_true', help='Label all vast outputs as vast 21')
    parser.add_argument("--merlin_only", action='store_true', help='Use only the post-merlin compiler graph representations for vast')
    parser.add_argument("--valid_only", action='store_true', help='Generate only valid designs for vast for regression estimation')
    parser.add_argument("--no_regen", action='store_true', help="Don't generate existing files")



    parser.add_argument("--output_folder", required=True, help='Output folder name in datasets')


    args = parser.parse_args()

    graph_config_name = None
    if args.graph_config_mayo:
        graph_config_name = GraphConfigNames.MAYO
    elif args.graph_config_galway:
        graph_config_name = GraphConfigNames.GALWAY
    elif args.graph_config_cavan:
        graph_config_name = GraphConfigNames.CAVAN
    elif args.graph_config_louth:
        graph_config_name = GraphConfigNames.LOUTH
    elif args.graph_config_kerry:
        graph_config_name = GraphConfigNames.KERRY
    elif args.graph_config_cork:
        graph_config_name = GraphConfigNames.CORK
    elif args.graph_config_limerick:
        graph_config_name = GraphConfigNames.LIMERICK

    kernelList = [] 

    if args.kernels_red:
        kernelList.append(KernelList.RED)
    if args.kernels_orange:
        kernelList.append(KernelList.ORANGE)
    if args.kernels_pink:
        kernelList.append(KernelList.PINK)


    if args.kernels_violet:
        kernelList.append(KernelList.VIOLET)


    if args.kernels_indigo:
        kernelList.append(KernelList.INDIGO)
    if args.kernels_ruby:
        kernelList.append(KernelList.RUBY)
    if args.kernels_sapphire:
        kernelList.append(KernelList.SAPPHIRE)


    if args.kernels_gold:
        kernelList.append(KernelList.GOLD)
    if args.kernels_silver:
        kernelList.append(KernelList.SILVER)
    if args.kernels_platinum:
        kernelList.append(KernelList.PLATINUM)
    if args.kernels_crystal:
        kernelList.append(KernelList.CRYSTAL)

    assert(graph_config_name is not None)
    assert(len(kernelList) > 0)

    generator = DatasetGenerator(args.dataset_folder, args.graph_compiler, args.inputs_folder, args.output_folder, graph_config_name, kernelList, args.combine_vast, args.all_vast_21, args.merlin_only, args.valid_only, args.no_regen)
    generator.generateData()
