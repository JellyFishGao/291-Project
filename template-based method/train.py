import pandas as pd
import logging

from seq2seq_model import Seq2SeqModel
#from sequence_labeling_model import SequenceLabelingModel
from transformers import TrainingArguments
import argparse


def template_based_pipeline(model_path='facebook/mbart-large-50', input_train_data='./data/BN-Bangla/5shots.csv',
                            input_dev_data='./data/BN-Bangla/dev.csv', best_model_dir="./outputs/best_model"):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_data = pd.read_csv(input_train_data, sep=',').values.tolist()
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    eval_data = pd.read_csv(input_dev_data, sep=',').values.tolist()
    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 50,
        "train_batch_size": 16,
        "num_train_epochs": 20,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 25,
        "manual_seed": 4,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": "./exp/template3",
        "early_stopping_metric_minimize": False,
        "early_stopping_metric": "eval_acc",
        "learning_rate": 2e-5,
        "best_model_dir": best_model_dir
    }

    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=model_path,
        args=model_args,
        # use_cuda=False,
    )

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    results = model.eval_model(eval_df)

    print(results)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Training pipeline for a template-based BART model.')

    # Add the model path argument
    parser.add_argument('--model_path', type=str, default="facebook/bart-base", help='path to the model')

    # Add the input train data argument
    parser.add_argument('--input_train_data', type=str, default='./data/BN-Bangla/50shots.csv',
                        help='path to input training data')

    # Add the input dev data argument
    parser.add_argument('--input_dev_data', type=str, default='./data/BN-Bangla/dev.csv',
                        help='path to input development data')

    # Add the best model directory argument
    parser.add_argument('--best_model_dir', type=str, default='./outputs/best_model',
                        help='path to directory to save best model')

    # Add the NER method argument
    parser.add_argument('--method_name', type=str, default='template_based',
                        help='method used for NER')

    # Add the dataset argument
    parser.add_argument('--dataset_name', type=str, default='multiconer',
                        help='dataset used for training')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the pipeline function with the command line arguments as input
    if args.method_name == 'template_based':
        template_based_pipeline(args.model_path, args.input_train_data, args.input_dev_data, args.best_model_dir)
    # elif args.method_name == 'sequence_labeling':
    #     sequence_labeling_pipeline(args.model_path, args.input_train_data, args.input_dev_data, args.best_model_dir,
    #                                args.dataset_name)
    else:
        print("please enter the correct method name")
