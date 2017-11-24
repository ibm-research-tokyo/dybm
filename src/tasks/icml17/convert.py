# (C) Copyright IBM Corp. 2017
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Script for converting dataset into numpy format.  Run this script
after downloading the datasets, as is explained in Readme.md.
"""

__author__ = "Takayuki Osogami"


from netCDF4 import Dataset
import numpy as np
import csv
import xlrd

if __name__ == "__main__":

    # here, specify where the downloaded data is located
    directory = "../../../data/"

    """ Sunspot data """

    filename = "monthly-sunspot-number-zurich-17.csv"

    npy_filename = filename[:-3] + "npy"
    print("\nConverting " + filename + " into " + npy_filename)

    with open(directory+filename, "r") as f:
        reader = csv.reader(f)
        data = list()
        for row in reader:
            if len(row) != 2:
                continue
            if row[0][4] != "-":
                continue
            data.append(float(row[1]))
    length = len(data)
    data = np.array(data).reshape((length,1))
    data.dump(directory+npy_filename)
    
    print("length, dimension: %d, %d" % data.shape)

    """ Price data """

    filename = "PET_PRI_GND_A_EPM0_PTE_DPGAL_W.xls"

    npy_filename = filename[:-3] + "npy"
    print("\nConverting " + filename + " into " + npy_filename)

    book = xlrd.open_workbook(directory+filename)
    sheet = book.sheet_by_index(2)
    data = list()
    for i in range(3,sheet.nrows):
        row = [sheet.cell(i,j).value for j in range(1,9)]
        data.append(row)

    length = 1223  # length of the time-series used in the ICML 2017 paper
    data = np.array(data[:length])
    data.dump(directory+npy_filename)

    print("length, dimension: %d, %d" % data.shape)
    
    
    """ Climate data """

    filename = "air.mon.anom.nc"

    npy_filename = filename[:-2] + "npy"
    print("\nConverting " + filename + " into " + npy_filename)
    f = Dataset(directory+filename, mode="r")

    length = f.variables["air"][:].data.shape[0]
    length = f.variables["air"][:].data.shape[0]
    count = np.product(f.variables["air"][:].data.shape[1:])

    # to ignore columns with missing values
    filled = f.variables["air"][:].data == f.variables["air"][:].fill_value
    never_filled = np.sum(filled, axis=0) == 0
    never_filled = never_filled.reshape(count)
    never_filled_idx = set(np.arange(1,count+1) * never_filled - 1)
    never_filled_idx.remove(-1)

    data = f.variables["air"][:].data.reshape((length,count))
    data = data[:,sorted(never_filled_idx)]
    data.dump(directory+npy_filename)

    print("length, dimension: %d, %d" % data.shape)

