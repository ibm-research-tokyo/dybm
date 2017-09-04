// (C) Copyright IBM Corp. 2015

package com.ibm.stdp.science;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;

import com.ibm.stdp.BitMapMusic;
import com.ibm.stdp.Network;
import com.ibm.stdp.TimeSeries;

/*
 * Analysis of the results of the Musician experiment
 * @author osogami
 */
public class AnalyzeMusician {
	
    public static void main(String[] args) {
	String directory = "Results/Musician/";
	
	BitMapMusic music = new BitMapMusic();
	final TimeSeries trainingData = music.getMusician();
	
	System.out.println("y = "+trainingData.toPython());		
	System.out.println("x = dict()");
	
	File dir = new File(directory);
	File[] files = dir.listFiles();
	for(File file : files){
	    String filename = file.getName();
	    if(!filename.endsWith(".bin")){
		continue;
	    }
	    String number = filename.split("param")[0].split("NN")[1];
	    int n = Integer.parseInt(number);
	    String result = directory+filename;
	    
	    Network ann;
	    
	    try {
		FileInputStream fis = new FileInputStream(result);
		ObjectInputStream is = new ObjectInputStream(fis);
		ann = (Network) is.readObject();
		is.close();
	    } catch (IOException | ClassNotFoundException e) {
		e.printStackTrace();
		return;
	    }
	    
	    // prediction
	    ann.store();
	    TimeSeries prediction = ann.freeRun(56,0);		
	    System.out.println("x["+n+"] = "+prediction.toPython());
	    ann.restore();
	}
    }
}
