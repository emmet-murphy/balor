from math import log2
import torch
import torch.types

from balorgnn.generate.encoders import OneHotEncoder

import numpy as np

from enum import Enum
class OutputConfigNames(Enum):
    DB4HLS = 0
    ML4ACCEL = 1
    POWERGEAR = 2
    VAST_18 = 3
    VAST_20 = 4
    VAST_21 = 5
    VAST_ALL = 6
    VAST_CUSTOM_18 = 7
    VAST_CUSTOM_20 = 8
    VAST_CUSTOM_21 = 9
    GNNDSE = 10

class OutputConfig():
    def __init__(self):
        self.log_normalized_metrics = []
        self.log_bias = {}
        self.log_scale_factors = {}
        self.affine_metrics = []
        self.affine_bias = {}
        self.one_hot_metrics = []
        self.one_hot_metric_encoder = {}

        self.custom_normalize = {}
        self.custom_unnormalize = {}

    def process_output_vector(self, kernel_data, i):
        normalized_values = []
        for metric in self.metrics:
            if metric in self.one_hot_metrics:
                value = kernel_data.get_values(metric, i)
                try:
                    one_hot_value = self.one_hot_metric_encoder[metric].encode(value)
                except Exception as e:     
                    modified_exception = ValueError(f"There was an error in one hot encoding {value} for {metric}")
                    raise modified_exception from e            
                normalized_values.extend(one_hot_value)
                continue
            metric_value = kernel_data.get_values(metric, i)
            # possibly a bad solution, but I've only ever seen 1 negative value
            # so for now it doesn't really matter
            metric_value = max(0, metric_value)
            normalized_value = self.normalize(metric, metric_value)
            normalized_values.append(normalized_value)

        # make 2D so that batching is easier
        normalized_values = [normalized_values]
        output = torch.tensor(normalized_values, dtype=torch.float32)
        return output


    def normalize(self, metric, value):
        if metric in self.custom_normalize:
            return self.custom_normalize[metric](value)
        
        if metric in self.log_normalized_metrics:
            bias = self.log_bias[metric]
            scale_factor = self.log_scale_factors[metric]
            log_value = log2(value + bias) - log2(bias)
            log_value_0_to_1 = log_value / scale_factor
            log_value_neg_1_to_1 = (log_value_0_to_1 * 2) - 1
            return log_value_neg_1_to_1
        
        assert(metric in self.max)
        
        if metric in self.affine_metrics:
            assert metric in self.min

            affine_val = value - self.min[metric]
            val_0_1 = affine_val / self.max[metric]
            return (2* val_0_1) - 1
        


        value_0_to_1 = value / self.max[metric]
        value_neg_1_to_1 = (value_0_to_1 * 2) - 1
        return value_neg_1_to_1

    def unnormalize(self, metric, value):

        if metric in self.custom_normalize:
            return self.custom_unnormalize[metric](value)
        if metric in self.log_normalized_metrics:
            bias = self.log_bias[metric]
            scale_factor = self.log_scale_factors[metric]
            value_0_to_1 = (value + 1) / 2
            log_value = (value_0_to_1 * scale_factor) + log2(bias)
            value = pow(2, log_value) - bias
            return value
        
        if metric in self.affine_metrics:
            assert metric in self.max
            assert metric in self.min
            val_0_1 = (value + 1) / 2
            affine_val = val_0_1 * self.max[metric]
            return affine_val + self.min[metric]
    

        assert(metric in self.max)
        value_0_to_1 = (value + 1) / 2
        value = value_0_to_1 * self.max[metric]
        return value
    
    def add_to_log_normalize(self, metric, bias):
        self.log_normalized_metrics.append(metric)
        self.log_bias[metric] = bias
        self.log_scale_factors[metric] = log2(self.max[metric] + bias) - log2(bias)

    def set_functions(self):
        pass

    def get_use_in_loss(self, i, kernel_data):
        assert(self.metrics)
        return torch.ones(len(self.metrics), dtype=torch.bool)

class OutputConfigDB4HLS(OutputConfig):
    def __init__(self):
        super().__init__()
        self.bias = 600
        self.metrics = ["LUTs", "FFs", "Latency", "BRAMs", "DSPs", "Clock"]
        self.max = {}
        self.max["LUTs"] = 350000
        self.max["FFs"] = 250000

        self.max["Latency"] = 58235500

        self.max["BRAMs"] = 64
        self.max["DSPs"] = 3500
        self.max["Clock"] = 13

        self.log_normalized_metrics.append("Latency")
        latency_bias = 600
        self.log_bias["Latency"] = latency_bias
        self.log_scale_factors["Latency"] = log2(self.max["Latency"] + latency_bias) - log2(latency_bias)

        self.log_normalized_metrics.append("LUTs")
        lut_bias = 300
        self.log_bias["LUTs"] = lut_bias
        self.log_scale_factors["LUTs"] = log2(self.max["LUTs"] + lut_bias) - log2(lut_bias)

        self.log_normalized_metrics.append("FFs")
        ff_bias = 600
        self.log_bias["FFs"] = ff_bias
        self.log_scale_factors["FFs"] = log2(self.max["FFs"] + ff_bias) - log2(ff_bias)

        self.log_normalized_metrics.append("BRAMs")
        bram_bias = 2
        self.log_bias["BRAMs"] = bram_bias
        self.log_scale_factors["BRAMs"] = log2(self.max["BRAMs"] + bram_bias) - log2(bram_bias)

        self.log_normalized_metrics.append("DSPs")
        dsp_bias = 100
        self.log_bias["DSPs"] = dsp_bias
        self.log_scale_factors["DSPs"] = log2(self.max["DSPs"] + dsp_bias) - log2(dsp_bias)

    def set_functions(self):
        pass


class OutputConfigML4ACCEL(OutputConfig):
    def __init__(self):
        super().__init__()
        self.metrics = ["Total LUTs", "Logic LUTs", "SRLs", "FFs", "RAMB36", "RAMB18", "DSP48 Blocks", "Latency", "Dynamic Power"]
        self.max = {}

        self.max["Total LUTs"] = 157308

        self.add_to_log_normalize("Total LUTs", 14000)


        self.max["Logic LUTs"] = 148078
        self.add_to_log_normalize("Logic LUTs", 14000)


        self.max["SRLs"] = 42764
        self.add_to_log_normalize("SRLs", 600)
    
        self.max["FFs"] = 184114
        self.add_to_log_normalize("FFs", 8000)


        self.max["Latency"] = 7366023
        self.add_to_log_normalize("Latency", 300000)

        self.max["Dynamic Power"] = 1823
        self.add_to_log_normalize("Dynamic Power", 2)

        self.max["RAMB36"] = 448
        self.add_to_log_normalize("RAMB36", 25)
        
        self.max["RAMB18"] = 448
        self.add_to_log_normalize("RAMB18", 25)

        self.max["DSP48 Blocks"] = 1024
        self.add_to_log_normalize("RAMB18", 14)

        # self.log_normalized_metrics.append("Latency")
        # latency_bias = 300000
        # self.log_bias["Latency"] = latency_bias
        # self.log_scale_factors["Latency"] = log2(self.max["Latency"] + latency_bias) - log2(latency_bias)

        # self.log_normalized_metrics.append("Dynamic Power")
        # power_bias = 2
        # self.log_bias["Dynamic Power"] = power_bias
        # self.log_scale_factors["Dynamic Power"] = log2(self.max["Dynamic Power"] + power_bias) - log2(power_bias)

class OutputConfigPowergear(OutputConfig):
    def __init__(self):
        super().__init__()
        self.metrics = ["Total Power", "Static Power", "Dynamic Power"]
        self.max = {}

        self.affine_metrics.append("Total Power")
        self.max["Total Power"] = 900000
        self.affine_bias["Total Power"] = 300000

        self.affine_metrics.append("Static Power")
        self.max["Static Power"] = 900000
        self.affine_bias["Static Power"] = 300000

        self.log_normalized_metrics.append("Dynamic Power")
        max_dynamic_power = 1331
        dynamic_bias = 0.5
        self.log_bias["Dynamic Power"] = dynamic_bias
        self.log_scale_factors["Dynamic Power"] = log2(max_dynamic_power + dynamic_bias) - log2(dynamic_bias)



class OutputConfigVast(OutputConfig):
    def __init__(self, join=False):
        super().__init__()

        base_metrics = ["LUTs", "FFs", "DSPs", "BRAMs", "Latency"]

        join_text = ""
        if join:
            join_text = " join"

        # self.set_types = [" valid", " all"]
        self.set_types = [" all"]
        self.metrics = []
        for metric in base_metrics:
            for set_type in self.set_types:
                specific_metric = f"{metric}{join_text}{set_type}"
                self.metrics.append(specific_metric)
                self.custom_normalize[specific_metric] = self.switch_unnormalize
                self.custom_unnormalize[specific_metric] = self.switch_unnormalize

        self.metrics.append(f"Valid{join_text}")
        self.metrics.append(f"Synthesized{join_text}")
        self.metrics.append(f"Oversized{join_text}")

        # self.max = {}
        # self.min = {}

        # for set_type in [" valid", " all"]:
        #     self.max[f"LUTs{join_text}{set_type}"] = 4
        #     self.max[f"FFs{join_text}{set_type}"] = 3
        #     self.max[f"DSPs{join_text}{set_type}"] = 8
        #     self.max[f"BRAMs{join_text}{set_type}"] = 3

        for set_type in [" valid", " all"]:
            self.custom_normalize[f"Latency{join_text}{set_type}"] = self.latency_to_perf
            self.custom_unnormalize[f"Latency{join_text}{set_type}"] = self.perf_to_latency
        
        for metric in ["Valid", "Synthesized", "Oversized"]:
            self.custom_normalize[f"{metric}{join_text}"] = self.switch_unnormalize
            self.custom_unnormalize[f"{metric}{join_text}"] = self.switch_unnormalize

    def set_functions(self):
        for join_text in ["", " join"]:
            for set_type in [" valid", " all"]:
                self.custom_unnormalize[f"Latency{join_text}{set_type}"] = self.perf_to_latency

       

    def switch_unnormalize(self, value):
        return value

    def zero_normalize(self, value):
        if value == 0:
            return -1
        return 1
    
    # def latency_to_perf(self, value):
    #     if value == 0:
    #         return -2
    #     removed_exp = (10 ** 7) / value
    #     perf = (log2(removed_exp) / 8) - 0.6
    #     return perf

    # def perf_to_latency(self, value):
    #     if value == -2:
    #         return torch.tensor(0)
    #     perf = value + 0.6
    #     perf = perf * 8
    #     removed_exp = 2 ** perf
    #     latency = (1 / removed_exp) * (10 ** 7)
    #     return latency

    def latency_to_perf(self, value):
        a = np.log(1e7/(value+1))/2
        # print(value, a)
        return a

    def perf_to_latency(self, value):
        return (1e7 / np.exp(2 * value)) - 1
    
    def get_use_in_loss(self, i, kernel_data):
        values = []
        valid_design = kernel_data.get_values("Valid", i)

        for metric in self.metrics:
            if "Valid" in metric or "Synthesized" in metric or "Oversized" in metric:
                values.append(True)
            elif " all" in metric:
                values.append(True)
            elif " valid" in metric:
                values.append(valid_design)
            else:
                print("Incorrect if formatting for loss", metric)
                quit()
        values = torch.tensor(values, dtype=torch.bool)
        return values
    

class OutputConfigGNNDSE(OutputConfig):
    def __init__(self):
        super().__init__()
        # self.metrics = ["Valid", "LUTs", "FFs", "DSPs", "BRAMs", "Latency"]
        self.metrics = ["LUTs", "FFs", "DSPs", "BRAMs", "Latency"]

        self.max = {}

        self.max["LUTs"] = 1
        # self.add_to_log_normalize("LUTs", 100000)

        self.max["FFs"] = 1
        # self.add_to_log_normalize("FFs", 100000)

        self.max["DSPs"] = 1
        # self.add_to_log_normalize("DSPs", 400)

        self.max["Latency"] = 7729501
        self.add_to_log_normalize("Latency", 200)

        self.max["BRAMs"] = 1
        # self.add_to_log_normalize("BRAMs", 10)

        # self.one_hot_metrics.append("Valid")
        # encoder = OneHotEncoder()
        # encoder.tags = [True, False]
        # encoder.fit()
        # self.one_hot_metric_encoder["Valid"] = encoder

