import glob
import os
import unittest

import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_something(self):
        train_main_path = '../data/Cattle-Disease-Classification/v2/'

        train_df = pd.DataFrame()

        def list_directories_recursively(path='.'):
            # The pattern '**/' will match all directories recursively
            out = []
            for dirname in glob.glob(f"{path}/**/", recursive=True):
                out.append(dirname)
                # yield dirname
            return out

        print("All directories recursively from current directory:")
        dirs = list_directories_recursively(train_main_path)
        print(dirs)

        all_classes = [
            'fmd',
            'lsd',
            'healthy',
        ]
        classes = []
        paths = []

        sub_dirs = list_directories_recursively(train_main_path)

        for one_class in all_classes:
            for one_sub in sub_dirs:
                if one_class.lower() in one_sub.lower():
                    for one_file in os.listdir(one_sub):
                        classes.append(one_class)
                        paths.append(one_sub + '/' + one_file)

        train_df['classname'] = classes
        train_df['path'] = paths

        # check empty
        emp_count = 0
        for dir in dirs:
            count = 0
            for x in paths:
                if dir in x:
                    count += 1
            if count < 5 :
                print(f'empty : {dir}')
                emp_count += 1
        print(f'empty count : {emp_count}')

        print(train_df.head())
        print(train_df.info())


if __name__ == '__main__':
    unittest.main()
