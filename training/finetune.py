import torch

from models.load_foldflow import load_model


def main():

    print("Starting fine-tuning...")

    model = load_model()

    # TODO:
    # load dataset
    # apply conditioning
    # training loop
    # save checkpoints


if __name__ == "__main__":
    main()
