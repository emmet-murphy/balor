

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile

from tabulate import tabulate

import os


def calculate_metrics(output_folder, epoch):
    csv_path = f"{output_folder}/val_{epoch}.csv"   

    df = pd.read_csv(csv_path)

    # inspired by the UCLA Vast papers (GNN-DSE, HARP, PROG SG, HLSYN, etc),
    # we choose the best model by RMSE on the normalized output of the model
    metrics = ["LUTs", "FFs", "DSPs", "BRAMs", "Latency", "Clock"]
    
    rmse = {}
    for metric in metrics:
        rmse[metric] = np.sqrt(np.mean((df[f"real {metric} normal"] - df[f"inferred {metric} normal"])**2))

    return rmse


def find_best_model(output_folder):
    min_sum = None

    if not os.path.isdir(output_folder):
        print(f"Results supposed to be located in {output_folder} were not found")
        quit()

    for epoch in range(10, 1000 + 10, 10):
        if not os.path.isfile(f"{output_folder}/val_{epoch}.csv"):
            continue

        sum = 0
        rmse = calculate_metrics(output_folder, epoch)
        for metric in rmse:
            sum = sum + rmse[metric]

        if min_sum is None or sum < min_sum:
            min_epoch = epoch
            min_sum = sum

    selection_metric = min_sum / len(rmse)

    print(f"Best model on validation set: Epoch {min_epoch} with a selection metric of {selection_metric}")
    print(f"Selection metric is average RMSE when inputs are scaled from -1 to 1")
    return min_epoch

def calculate_perc_error(output_folder, epoch):

    if not os.path.isfile(f"{output_folder}/test_{epoch}.csv"):
        raise RuntimeException("Test file for best epoch did not exist")

    df = pd.read_csv(f"{output_folder}/test_{epoch}.csv")

    for metric in ["LUTs", "FFs", "DSPs", "BRAMs", "Latency", "Clock"]:
        df_temp = df.copy()

        # no perc error if real is 0
        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]

        perc_error = (abs(df_temp[f"real {metric}"] - df_temp[f"inferred {metric}"]) / df_temp[f"real {metric}"]) * 100
        perc_error = (abs(df_temp[f"real {metric}"] - df_temp[f"inferred {metric}"]) / df_temp[f"real {metric}"]) * 100
        
        mean_perc_error = perc_error.mean()

        print(f"Mean Percent Error for {metric}: {mean_perc_error}")

def make_pdf(report_name, output_folder, epoch):

    if not os.path.isfile(f"{output_folder}/test_{epoch}.csv"):
        raise RuntimeException("Test file for best epoch did not exist")

    df = pd.read_csv(f"{output_folder}/test_{epoch}.csv")


    # Step 3: Generate the PDF report
    class PDFReport(FPDF):
        def header(self):
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, f"Balor Evaluation Report: {report_name} at Epoch {epoch}", 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


    # Create an instance of the PDF class and add a page
    pdf = PDFReport()
    pdf.add_page()

    pdf.ln(5)
    pdf.set_font("Courier", size=8)

    for metric in ["LUTs", "FFs", "Latency", "Clock", "DSPs", "BRAMs"]:
        table = []
        headers = []

        headers.append(f"{metric} Median Error ")
        headers.append(f"{metric} Mean Error ")
        headers.append(f"{metric} 95th Percentile Error ")

        table.append(headers)

        df_temp = df.copy()
        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]
        perc_error = (abs(df_temp[f"real {metric}"] - df_temp[f"inferred {metric}"]) / df_temp[f"real {metric}"]) * 100


        mean = perc_error.mean().round(1)
        median = perc_error.median().round(1)
        top = np.percentile(perc_error, 95).round(1)

        values = [median, mean, top]

        table.append(values)

        out = tabulate(table, headers='firstrow', tablefmt="grid")

        pdf.multi_cell(0, 3, out, align="C")

        pdf.ln(10)  

    pdf.add_page()

    for i, metric in enumerate(["LUTs", "FFs", "Latency", "Clock", "DSPs", "BRAMs"]):
        # Step 2: Generate a scatter plot of Column_X vs Column_Y
        plt.figure(figsize=(8, 5))

        real_cut = np.percentile(df[f"real {metric}"], 90)
        inferred_cut = np.percentile(df[f"inferred {metric}"], 90)

        min_val = min(df[f"real {metric}"].min(), df[f"inferred {metric}"].min())
        max_val = max(real_cut, inferred_cut)
        plt.plot([min_val, max_val], [min_val, max_val],  lw=1.5, color="orange", linestyle="--")


        plt.scatter(df[f"real {metric}"], df[f"inferred {metric}"], alpha=0.7, s=5)
        
        if metric == "Clock":
            metric = "Clock Period (ns)"

        if metric == "Latency":
            metric = "Latency (cycles)"
        
        plt.title(f"Scatter Plot cut at 90th percentile: Real {metric} vs. Inferred {metric}")
        # plt.title(f"Real {metric} vs. Inferred {metric}")

        plt.xlabel(f"Real {metric}")
        plt.ylabel(f"Inferred {metric}")
        diff = max_val - min_val
        plt.ylim([min_val - (diff*0.05), max_val + (diff*0.05)])
        plt.xlim([min_val- (diff*0.05), max_val + (diff*0.05)])


        # Save the plot to a temporary file
        scatter_plot_path = tempfile.mktemp(".png")
        plt.tight_layout()
        plt.savefig(scatter_plot_path)
        plt.close()

        # Step 4: Insert the scatter plot image into the PDF
        pdf.image(scatter_plot_path, x=10, y=20 + (130*(i%2)), w=180)

        if i == 1 or i == 3:
            pdf.add_page()

    pdf.add_page()


    for i, metric in enumerate(["LUTs", "FFs", "Latency", "Clock", "DSPs", "BRAMs"]):
        # Step 2: Generate a scatter plot of Column_X vs Column_Y
        plt.figure(figsize=(8, 5))

        min_val = min(df[f"real {metric}"].min(), df[f"inferred {metric}"].min())
        max_val = max(df[f"real {metric}"].max(), df[f"inferred {metric}"].max())
        plt.plot([min_val, max_val], [min_val, max_val],  lw=1.5, color="orange", linestyle="--")


        plt.scatter(df[f"real {metric}"], df[f"inferred {metric}"], alpha=0.7, s=5)
        if metric == "Clock":
            metric = "Clock Period (ns)"

        if metric == "Latency":
            metric = "Latency (cycles)"

        plt.title(f"Scatter Plot Full: Real {metric} vs. Inferred {metric}")
        plt.xlabel(f"Real {metric}")
        plt.ylabel(f"Inferred {metric}")


        # Save the plot to a temporary file
        scatter_plot_path = tempfile.mktemp(".png")
        plt.tight_layout()
        plt.savefig(scatter_plot_path)
        plt.close()

        # Step 4: Insert the scatter plot image into the PDF
        pdf.image(scatter_plot_path, x=10, y=20 + (130*(i%2)), w=180)

        if i == 1 or i == 3:
            pdf.add_page()

    pdf.add_page()


    for i, metric in enumerate(["LUTs", "FFs", "Latency", "Clock", "DSPs", "BRAMs"]):
        plt.figure(figsize=(8, 5))

        df_temp = df.copy()

        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]



        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]


        df_temp['percent_error'] = (abs(df_temp[f"inferred {metric}"] - df_temp[f"real {metric}"]) / df_temp[f"real {metric}"]) * 100

        top = np.percentile(df_temp['percent_error'], 95)
        df_temp = df_temp[df_temp['percent_error'] < top]


        plt.hist(df_temp['percent_error'], bins=1000, density=True)        


        if metric == "Clock":
            metric = "Clock Period (ns)"

        if metric == "Latency":
            metric = "Latency (cycles)"
        
        plt.title(f"Percent Error Histogram: Real {metric} vs. Inferred {metric}")
        # plt.title(f"Real {metric} vs. Inferred {metric}")

        plt.xlabel(f"Real {metric}")
        plt.ylabel(f"Inferred {metric}")

        plt.xlim([0, 15])

        # Save the plot to a temporary file
        plot_path = tempfile.mktemp(".png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Step 4: Insert the scatter plot image into the PDF
        pdf.image(plot_path, x=10, y=20 + (130*(i%2)), w=180)

        if i == 1 or i == 3:
            pdf.add_page()

    pdf.add_page()



    for i, metric in enumerate(["LUTs", "FFs", "Latency", "Clock", "DSPs", "BRAMs"]):
        plt.figure(figsize=(8, 5))

        df_temp = df.copy()

        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]



        df_temp = df_temp[df_temp[f"real {metric}"] > 0.001]


        df_temp['percent_error'] = (abs(df_temp[f"inferred {metric}"] - df_temp[f"real {metric}"]) / df_temp[f"real {metric}"]) * 100

        # Calculate the histogram
        counts, bin_edges = np.histogram(df_temp['percent_error'], bins=1000, density=True)

        # Calculate the cumulative distribution (CDF) from the histogram
        cdf = np.cumsum(counts) * np.diff(bin_edges)
        cdf = np.insert(cdf, 0, 0)  # Include the starting point (0 probability)

        plt.plot(bin_edges, cdf, marker='.', linestyle='-')

        plt.axhline(y=1, color='r', linestyle='--')


        if metric == "Clock":
            metric = "Clock Period (ns)"

        if metric == "Latency":
            metric = "Latency (cycles)"
        
        plt.title(f"Percent Error Numerical CDF: Real {metric} vs. Inferred {metric}")
        # plt.title(f"Real {metric} vs. Inferred {metric}")

        plt.xlabel(f"Real {metric}")
        plt.ylabel(f"Inferred {metric}")

        plt.xlim([0, 30])


        # Save the plot to a temporary file
        plot_path = tempfile.mktemp(".png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # Step 4: Insert the scatter plot image into the PDF
        pdf.image(plot_path, x=10, y=20 + (130*(i%2)), w=180)

        if i == 1 or i == 3:
            pdf.add_page()




    # Step 5: Output the PDF to a file
    output_pdf_path = f"reports/{report_name}_report.pdf"
    pdf.output(output_pdf_path)

    print(f"PDF report generated successfully: {output_pdf_path}")



def report_single(dataset, id=0):
    output_folder = f"balorgnn/outputs/results/{dataset}/limerick/snake/{id}"
    epoch = find_best_model(output_folder)
    calculate_perc_error(output_folder, epoch)
    make_pdf(dataset, output_folder,epoch)

