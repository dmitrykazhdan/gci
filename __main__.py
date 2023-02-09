from experiments.synthetic_experiments.experiment_runner import run_synthetic_experiments
from experiments.torchdrug_experiments.experiment import run_torchdrug_dataset_experiment

def main():
    # Uncomment for synthetic dataset
    # run_synthetic_experiments()

    # Uncomment for torchdrug experiment
    run_torchdrug_dataset_experiment()


if __name__ == "__main__":
    main()