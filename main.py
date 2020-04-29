import json
import algorithms
import genetic_feature_selection
import relief as relief
import data_loader as loader
import algorithms as algs
import pymrmr as mrmr
from mrmrWrapper import MrmrWrapper
from reliefWrapper import ReliefWrapper
from fcbfWrapper import FcbfWrapper
from IWFS import IwfsWrapper
from K_Nearest_Neighbors import K_Nearest_Neighbors
from Decision_Tree import Decision_Tree
from Support_Vector_Machine import Support_Vector_Machine


relief_key = "relief"
mrmr_key = "mrmr"
iwfs_key = "iwfs"
fcbf_key = 'fcbf'

genetic_key = 'genetic'
genetic_generations_num = 'numberOfGenerations'
genetic_best_num = 'bestChromosomes'
genetic_chromosomes_num = 'chromosomesNum'
genetic_rand_num = 'randChromosomesNum'
genetic_crossover_num = 'crossoverNum'
genetic_operator_prob = 'operatorsProbability'

arguments_path = 'params.txt'
fcbf_threshhold_key = 'threshhold'
data_path_key = 'path'
num_of_bins_key = "numberOfBins"
num_features_to_return_key = "returnNumberOfFeatures"
mrmr_type_key = "type"
num_of_neighbors_key = "numberOfNeighbors"

def main():
    with open(arguments_path, 'r') as json_metadata:
        json_str = ' '.join(json_metadata.readlines())
        json_file = json.loads(json_str)
        algs_dict = dict()
        algs_list = []
        file_path = get_json_value(json_file, data_path_key)
        number_of_bins = get_json_value(json_file, num_of_bins_key)
        features_vectors, class_vector, features_names, class_name = get_data_parameters(file_path, number_of_bins)

        algs_dict['mrmr'] = prepare_mrmr_alg(json_file)
        algs_dict['relief'] = prepare_relief_alg(json_file, class_vector)
        algs_dict['fcbf'] = prepare_fcbf_alg(json_file, class_vector)
        algs_dict['iwfs'] = prepare_iwfs_alg(json_file, class_vector, class_name, features_names)

        algs_data_dict = run_algs(algs_dict, features_vectors)

        feed_to_models(algs_data_dict)


def feed_to_models(algs_data_dict, class_vector, features_names, class_name):

    svm = Support_Vector_Machine(algs_data_dict['fcbf'], class_vector, features_names, class_name)
    svm.get_mean_accuarcy_by_cross_validation()

    return


def prepare_iwfs_alg(json_file, class_vector, class_name, features_names):
    iwfs_json = get_json_value(json_file, iwfs_key)
    iwfs_num_features_to_return = get_json_value(json_file, num_features_to_return_key)
    return IwfsWrapper(class_vector, len(features_names), features_names, class_name)

def prepare_fcbf_alg(json_file, class_vector):
    fcbf_json = get_json_value(json_file, fcbf_key)
    fcbf_threshhold = get_json_value(fcbf_json, fcbf_threshhold_key)
    return FcbfWrapper(class_vector, fcbf_threshhold)

def run_algs(algs_dict, data):
    algs_data_dict = dict()
    # algs_data_dict['mrmr'] = algs_dict['mrmr'].fit_data(data)
    # algs_data_dict['relief'] = algs_dict['relief'].fit_data(data)
    # algs_data_dict['fcbf'] = algs_dict['fcbf'].fit_data(data)
    algs_data_dict['iwfs'] = algs_dict['iwfs'].fit_data(data)

    return algs_data_dict

def prepare_relief_alg(json_file, class_vector):
    relief_json = get_json_value(json_file, relief_key)
    relief_num_neighbors = get_json_value(relief_json, num_of_neighbors_key)
    relief_num_features_to_keep = get_json_value(json_file, num_features_to_return_key)
    return ReliefWrapper(class_vector, relief_num_neighbors, relief_num_features_to_keep)

def prepare_mrmr_alg(json_file):
    mrmr_json = get_json_value(json_file, mrmr_key)
    mrmr_type = get_json_value(mrmr_json, mrmr_type_key)
    mrmr_num_features = get_json_value(json_file, num_features_to_return_key)
    return MrmrWrapper(mrmr_type, mrmr_num_features)

def create_fcbf_alg(args_json):
    threshhold = get_json_value(args_json, fcbf_threshhold_key)
    return algorithms.fcbf(threshhold=threshhold)

def create_genetic_alg(args_json):
    generations_num = get_json_value(args_json, genetic_generations_num)
    best_generations_num = get_json_value(args_json, genetic_best_num)
    chromosome_num = get_json_value(args_json,  genetic_chromosomes_num)
    rand_chromoomes_num = get_json_value(args_json, genetic_rand_num)
    crossover_num = get_json_value(args_json, genetic_crossover_num)
    operator_prob = get_json_value(args_json, genetic_operator_prob)
    return genetic_feature_selection.GeneticSelector(
        estimator='get etimator',
        num_of_generations=generations_num,
        num_of_chromosomes=chromosome_num,
        num_best_chromosomes=best_generations_num,
        num_rand_chromosomes=rand_chromoomes_num,
        num_crossover_children=crossover_num,
        operator_probability=operator_prob,
        features_names='get feature names',
        class_vector='get class vector'
    )

def create_iwfs_alg():
    return

def get_json_value(json_file, key):
    try:
        return json_file[key]
    except ValueError:
        print("cant find key %s", key)
        raise ValueError('cannot find given key %s', key)


def load_data_as_df(path, num_of_bins):
    return loader.data_processing(path, sep=',', number_of_bins=num_of_bins)


def get_data_parameters(path, num_of_bins):
    process = load_data_as_df(path, num_of_bins)
    return process.prepare_data()



if __name__ == '__main__':
    main()