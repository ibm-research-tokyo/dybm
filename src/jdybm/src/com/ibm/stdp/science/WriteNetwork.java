// (C) Copyright IBM Corp. 2015
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package com.ibm.stdp.science;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;

import com.ibm.stdp.Network;

/*
 * Writing out the DyBM after training
 * @author osogami
 */
public class WriteNetwork {
    public static void main(String[] args) {
	String directory = "Results/SCIENCE/Single3b/";
	File dir = new File(directory);
	File[] files = dir.listFiles();
	
	for(File file : files){
	    String filename = file.getName();
	    System.out.println(filename);
	    if(filename.startsWith("NN") && filename.endsWith(".bin")){
		Network ann;
		try {
		    FileInputStream fis = new FileInputStream(file);
		    ObjectInputStream is = new ObjectInputStream(fis);
		    ann = (Network) is.readObject();
		    is.close();
		} catch (IOException | ClassNotFoundException e) {
		    e.printStackTrace();
		    return;
		}
		String[] split = filename.split("\\.");
		String outname = directory + split[0] + ".csv";
		System.out.println(outname);
		try {
		    File f = new File(outname);
		    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(f)));
		    pw.println(ann.toCSV());
		    pw.close();
		} catch (IOException e) {					
		    e.printStackTrace();
		}
	    }
	}
    }
}
