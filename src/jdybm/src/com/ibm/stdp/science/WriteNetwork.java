// (C) Copyright IBM Corp. 2015

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
