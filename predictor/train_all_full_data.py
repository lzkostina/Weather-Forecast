from train_models import train_all_full
def main():
    """
    Train all models on the full dataset.
    """
    print("Initializing training on the full dataset...")
    train_all_full()
    print("All models have been successfully trained on the full dataset.")

if __name__ == "__main__":
    main()
