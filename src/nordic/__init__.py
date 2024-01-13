import argparse

__version__ = '0.1.0'

def main():
    parser = argparse.ArgumentParser(description='nordic')
    
    parser.add_argument("arg1", help='arg1')
    parser.add_argument("--arg2", help='arg2')
    
    args = parser.parse_args()
    
    arg1_value = args.arg1
    arg2_value = args.arg2
    
    print(arg1_value, arg2_value)
    

if __name__ == "__main__":
    main()