import argparse
import pickle


from in2004_ew_shingle import NewPolicyRunner


def main():

    parser = argparse.ArgumentParser(description="Run NewPolicy.")
    parser.add_argument("--dataset", nargs="?", default="in-2004", help="Dataset name")
    parser.add_argument("--counts", type=int, default=100, help="Number of iterations of node-mergings.")
    parser.add_argument("--filters1", type=int, default=64, help="Filters (neurons) in 1st convolution.")
    parser.add_argument("--filters2", type=int, default=32, help="Filters (neurons) in 2nd convolution.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--bad_counter", type=int, default=1, help="Value of bad counts to tolerate.")
    args = parser.parse_args()


    executer = NewPolicyRunner(args)
    executer.fit()


if __name__ == "__main__":
    main()
