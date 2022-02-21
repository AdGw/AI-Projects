from format_raw_to_clean import format_data
from remove_duplicates import remove_duplicates
from split_3_models import split_models
from generate_csv import generate
from ev_models import evaluate_models
from generate_dict import generate_dict
from create_csv import create
from ev_models_features import evaluate_models_features
from create_website_list import create_list
from create_vectorized import create_vectorize
from create_X_features import create_X
from create_Y_features import create_Y
from ev_joblib_models import evaluate_joblib_models


def machine_learning():
    format_data()
    print("format data")
    remove_duplicates()
    print("remove duplicates")
    split_models()
    print("split models")
    generate()
    print("generate csv")
    evaluate_models()
    print("evaluate models")


def machine_learning_from_features():
    generate_dict()
    print("generate dict")
    create()
    print("create csv")
    evaluate_models_features()


def machine_learning_joblib():
    create_list()
    print("joblib list create")
    create_vectorize()
    print("create vectorize")
    create_X()
    print("create X")
    create_Y()
    print("create Y")
    evaluate_joblib_models()
    print("evaluate joblib models")


if __name__ == '__main__':
    machine_learning()
    machine_learning_from_features()
    machine_learning_joblib()
