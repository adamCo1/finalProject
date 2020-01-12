import numpy as np
import os
import tkinter as tk
from tkinter import filedialog as fd
from enum import Enum
import genetic_feature_selection as genetic_selector
import data_loader as dl
import Mutual_Information_Estimator as mi
import csv
import pandas as pd

class Browse(tk.Frame):
    """ Creates a frame that contains a button when clicked lets the user to select
    a file and put its filepath into an entry.
    """

    def __init__(self, master, initialdir='', filetypes=()):
        super().__init__(master)
        self.dataset_path = tk.StringVar()
        self.num_of_discritization_bins = tk.StringVar()
        self.num_of_generations = tk.StringVar()
        self.num_of_chromosomes = tk.StringVar()
        self.num_best_chromosomes = tk.StringVar()
        self.num_rand_chromosomes = tk.StringVar()
        self.num_crossover_children = tk.StringVar()
        self.operator_probability = tk.StringVar()
        # self.dataset_path = tk.StringVar()
        self._initaldir = initialdir
        self._filetypes = filetypes
        self._create_widgets()
        self._display_widgets()

    def _create_widgets(self):
        self.entry_dataset_path = tk.Entry(self, textvariable=self.dataset_path, font=("bold", 10))
        self.num_of_discritization_bins = tk.Entry(self, textvariable=self.num_of_discritization_bins, font=("bold", 10))
        self.num_of_generations = tk.Entry(self, textvariable=self.num_of_generations, font=("bold", 10))
        self.num_of_chromosomes = tk.Entry(self, textvariable=self.num_of_chromosomes, font=("bold", 10))
        self.num_best_chromosomes = tk.Entry(self, textvariable=self.num_best_chromosomes, font=("bold", 10))
        self.num_rand_chromosomes = tk.Entry(self, textvariable=self.num_rand_chromosomes, font=("bold", 10))
        self.num_crossover_children = tk.Entry(self, textvariable=self.num_crossover_children, font=("bold", 10))
        self.operator_probability = tk.Entry(self, textvariable=self.operator_probability, font=("bold", 10))
        self._button_prediction = tk.Button(self, text="Browse dataset",bg="blue",fg="white", command=self.browsePredict)
        self._label_p1=tk.Label(self, text="dataset path")
        self._label_p2=tk.Label(self, text="number of bins")
        self._label_p3=tk.Label(self, text="number of generations")
        self._label_p4=tk.Label(self, text="population size")
        self._label_p5=tk.Label(self, text="number of best chromosomes")
        self._label_p6=tk.Label(self, text="number of random chromosomes")
        self._label_p7=tk.Label(self, text="number of crossover childs")
        self._label_p8=tk.Label(self, text="operator probabilty")





        # a=self._entry_prediction
        # self._entry_model = tk.Entry(self, textvariable = self.modelPath, font = ("bold", 10))
        # self._button_prediction = tk.Button(self, text="Browse Model File",bg="blue",fg="white", command=self.browsePredict)
        # self._button_model = tk.Button(self, text="Browse Images Folder",bg="blue",fg="white", command=self.browseModel)
        self._select=tk.Button(self,text="Select",bg="blue",fg="white", command=self.select)
        self._label=tk.Label(self, text="Genetic Feature Selection.", bg="blue", fg="black",height=3, font=("bold", 14))



    def _display_widgets(self):
        self._label.pack(fill='y')
        self._label_p1.pack(fill='y')
        self.entry_dataset_path.pack(fill='x', expand=True)
        self._button_prediction.pack(fill='x', expand=True)
        self._label_p2.pack(fill='y')
        self.num_of_discritization_bins.pack(fill='x', expand=True)
        self._label_p3.pack(fill='y')
        self.num_of_generations.pack(fill='x', expand=True)
        self._label_p4.pack(fill='y')
        self.num_of_chromosomes.pack(fill='x', expand=True)
        self._label_p5.pack(fill='y')
        self.num_best_chromosomes.pack(fill='x', expand=True)
        self._label_p6.pack(fill='y')
        self.num_rand_chromosomes.pack(fill='x', expand=True)
        self._label_p7.pack(fill='y')
        self.num_crossover_children.pack(fill='x', expand=True)
        self._label_p8.pack(fill='y')
        self.operator_probability.pack(fill='x', expand=True)
        # self._button_model.pack(fill='y')
        # self._entry_prediction.pack(fill='x', expand=True)
        # self._entry_model.pack(fill='x', expand=True)
        self._select.pack(fill='y')

    # def retrieve_input(self):
    #     #str1 = self._entry.get()
    #     # a=a.replace('/','//')
    #     print (str1)


    def select(self):
        newwin = tk.Toplevel(root)
        newwin.geometry("500x500")
        label = tk.Label(newwin, text="Best subsets", bg="blue", fg="white",height=3, font=("bold", 14))
        label.pack()
        entry_dataset_path = self.entry_dataset_path.get()
        num_of_discritization_bins = self.num_of_discritization_bins.get()
        num_of_generations = self.num_of_generations.get()
        num_of_chromosomes = self.num_of_chromosomes.get()
        num_best_chromosomes = self.num_best_chromosomes.get()
        num_rand_chromosomes = self.num_rand_chromosomes.get()
        num_crossover_children = self.num_crossover_children.get()
        operator_probability = self.operator_probability.get()
        try:

        # model = prepare_network_from_file(self.modelPath.get())
        # predictions = predict_from_folder(model,self.filepath.get())
        #     data_vector, target_vector = dataset.data, dataset.target
        #     features = dataset.feature_names
        #     for i in range(20):
            data_proccessing = dl.data_processing(dataset_path=entry_dataset_path, sep=",", number_of_bins=int(num_of_discritization_bins))
            features_values, class_values = data_proccessing.prepare_data()
            feature_names = features_values.columns.values
        # feature_names.append(class_values.name)
        # feature_names = pd.DataFrame(feature_names)

            mi_estimator = genetic_selector.Mutual_Information_Estimator(features_values, class_values, feature_names)
            selector = genetic_selector.GeneticSelector(estimator =mi_estimator,
                                           num_of_generations = int(num_of_generations),
                                           num_of_chromosomes = int(num_of_chromosomes),
                                           num_best_chromosomes = int(num_best_chromosomes),
                                           num_rand_chromosomes = int(num_rand_chromosomes),
                                           num_crossover_children = int(num_crossover_children),
                                           features_names = feature_names,
                                           operator_probability = float(operator_probability))
            selector.evolve(features_values, class_values)
            best_features = selector.best_features
        #         print(i,best_features)
        #

            T = tk.Text(newwin, height=25, width=60)
            for list in best_features:
                T.insert('end',str(list)+"\n")
                T.pack()
        except :
            raise ValueError('please insert valid parameters')

        #then show it on screen
        newwin.mainloop()

    def browseModel(self):

        self.dataset_path.set(fd.askdirectory())

    def browsePredict(self):
        self.dataset_path.set(fd.askopenfilename(initialdir=self._initaldir))



if __name__ == '__main__':
    root = tk.Tk()
    labelfont = ('times', 10, 'bold')
    root.geometry("500x500")
    # filetypes = (
    #     ('Image File', '*.csv')
    # )

    file_browser = Browse(root, initialdir="\\")
    file_browser.pack(fill='y')
    root.mainloop()