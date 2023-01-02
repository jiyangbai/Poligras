import argparse
import pickle


from in2004_ew_shingle import NewPolicyRunner


def main():

    parser = argparse.ArgumentParser(description="Run NewPolicy.")
    parser.add_argument("--dataset", nargs="?", default="in-2004", help="Dataset name")
    parser.add_argument("--counts", type=int, default=1, help="Number of iterations in each epoch. Default is 1.")
    parser.add_argument("--filters1", type=int, default=64, help="Filters (neurons) in 1st convolution. Default is 64.")
    parser.add_argument("--filters2", type=int, default=32, help="Filters (neurons) in 2nd convolution. Default is 32.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. Default is 128.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability. Default is 0.0.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay. Default is 0.0.")
    parser.add_argument("--bad_counter", type=int, default=1, help="Value of bad counts to tolerate. Default is 5.")
    args = parser.parse_args()


    executer = NewPolicyRunner(args)
    executer.fit()


if __name__ == "__main__":
    main()
