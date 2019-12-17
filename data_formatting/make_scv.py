import sys


def main(argv):
    if len(argv)!=3:
        print("usage: python3 %s <file_column1> <file_column2>"%argv[0])
        sys.exit(42)

    print("\"before\",\"after\"")
    for l1, l2 in zip(open(argv[1], "r"), open(argv[2], "r")):
        l1 = l1[:-1].replace('"', '""')
        l2 = l2[:-1].replace('"', '""')
        print("\"%s\",\"%s\""%(l1,l2))


if __name__ == "__main__":
    main(sys.argv)
