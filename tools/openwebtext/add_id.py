# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import os
import time

"""
This code adds id to each json object in a json file. User can add prefix
to the ids.
"""

if __name__ == '__main__':

    print('parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default=None, help='Input'\
        ' json file where id needs to be added')
    parser.add_argument('--output-file', type=str, default=None, help=\
        'Output file name with id')
    parser.add_argument('--id-prefix', type=str, default=None, help=\
        'Id prefix')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Log interval')
    args = parser.parse_args()

    print('Adding ids to dataset ...')

    f_input = open(args.input_file, 'r', encoding='utf-8')
    f_output = open(args.output_file, 'wb')

    unique_ids = 1
    start_time = time.time()
    for row in f_input:
        each_row = json.loads(row)
        adlr_id_string = args.id_prefix + '-{:010d}'.format(int(unique_ids))
        each_row['adlr_id'] = adlr_id_string
        myjson = json.dumps(each_row, ensure_ascii=False)

        f_output.write(myjson.encode('utf-8'))
        f_output.write('\n'.encode('utf-8'))

        if unique_ids % args.log_interval == 0:
            print('    processed {:9d} documents in {:.2f} seconds ...'.format( \
                    unique_ids, time.time() - start_time), flush=True)

        unique_ids += 1

    # Close the file.
    f_input.close()
    f_output.close()
    
    print('done :-)', flush=True)
