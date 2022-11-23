#!/usr/bin/env python3
import argparse
import os


def main(dataset_name,output):
    _o_d = os.path.abspath(output)
    os.makedirs(_o_d,exist_ok=True)
    os.system(f"vessl dataset download {dataset_name} train {os.path.join(_o_d,'train')}")
    os.system(f"vessl dataset download {dataset_name} validation {os.path.join(_o_d,'validation')}")
    os.system(f"vessl dataset download {dataset_name} test {os.path.join(_o_d,'test')}")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='download dataset')
    parser.add_argument('-i','--dataset_name',
                        default='2016LowDoseCTGrandChallenge',
                        choices=['2016LowDoseCTGrandChallenge','2016LowDoseCTGrandChallenge-lite'],
                        help='Name of dataset')
    parser.add_argument('-o','--output',default='./',help='Output directory')
    args = parser.parse_args()
    main(dataset_name=args.dataset_name,output=args.output)
    
