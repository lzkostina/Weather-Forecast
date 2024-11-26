from train_models import train_all  # Import the function from your training module

def main():
    """
    Train all models on the partial dataset.
    """
    print("Initializing training on the partial dataset...")
    train_all()
    print("All models have been successfully trained on the partial dataset.")

if __name__ == "__main__":
    main()