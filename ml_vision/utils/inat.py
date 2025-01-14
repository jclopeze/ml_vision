import numpy as np
import pandas as pd
from struct import unpack

inat_input_images = ['original', 'large', 'medium', 'small', 'thumb', 'square']

def read_burst(srow,
               burst,
               cols_names,
               origin_csv,
               dest_csv,
               filter_col,
               filter_vals,
               dtypes,
               cols_to_write,
               usecols=None,
               delimiter=','):
    header = srow == 1
    df = pd.read_csv(origin_csv, skiprows=srow, nrows=burst,
                     names=cols_names, delimiter=delimiter, dtype=dtypes, usecols=usecols)
    df = df.drop(df[~df[filter_col].str.lower().isin(filter_vals)].index)
    df.to_csv(dest_csv, mode='a', header=header, columns=cols_to_write, index=False)
    del df


def read_csv_in_burst(origin_csv,
                      dest_csv,
                      cols_names,
                      filter_col,
                      filter_vals,
                      dtypes,
                      cols_to_write,
                      usecols=None,
                      burst_size=2000000,
                      delimiter=','):

    srow = 1
    nrows = get_num_rows_in_csv(origin_csv)

    while srow < nrows:
        read_burst(srow, burst_size, cols_names, origin_csv, dest_csv,
                   filter_col, filter_vals, dtypes, cols_to_write, usecols=usecols,
                   delimiter=delimiter)
        srow += burst_size
    read_burst(srow, nrows, cols_names, origin_csv, dest_csv,
               filter_col, filter_vals, dtypes, cols_to_write, usecols=usecols,
               delimiter=delimiter)


def get_num_rows_in_csv(origin_csv):
    nrows = 0
    chunk = 1024*1024   # Process 1 MB at a time.
    f = np.memmap(origin_csv)
    nrows = sum(np.sum(f[i:i+chunk] == ord('\n')) for i in range(0, len(f), chunk))
    del f

    return nrows


def get_corrupted_images(image, bads):
    class JPEG:
        def __init__(self, image_file):
            with open(image_file, 'rb') as f:
                self.img_data = f.read()

        def decode(self):
            data = self.img_data
            while(True):
                marker, = unpack(">H", data[0:2])
                if marker == 0xffd8:
                    data = data[2:]
                elif marker == 0xffd9:
                    return
                elif marker == 0xffda:
                    data = data[-2:]
                else:
                    lenchunk, = unpack(">H", data[2:4])
                    data = data[2+lenchunk:]
                if len(data)==0:
                    break

    img = JPEG(image)
    try:
        img.decode()
    except:
        bads.append(image)