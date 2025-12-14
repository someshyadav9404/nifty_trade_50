import os
import subprocess

def run_step(command):
    print(f"\n===============================")
    print(f" Running: {command}")
    print(f"===============================\n")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error running: {command}")
        exit(1)
    print(f"✅ Completed: {command}\n")


def main():
    # Step 1: Fetch fresh data
    run_step("python src/data_pipeline/fetch_data.py")

    # Step 2: Process & feature engineering + labeling
    run_step("python src/data_pipeline/process_data.py")

    # Step 3: Train model
    run_step("python src/model_training/train_model.py")

    # Step 4: Run backtest
    run_step("python src/backtesting/backtest.py")

    print("\n🎉 All pipeline steps successfully completed.\n")

if __name__ == "__main__":
    main()
