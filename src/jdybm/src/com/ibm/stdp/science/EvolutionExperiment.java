// (C) Copyright IBM Corp. 2015

package com.ibm.stdp.science;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import com.ibm.stdp.BitMapImage;
import com.ibm.stdp.Network;
import com.ibm.stdp.Parameter;
import com.ibm.stdp.TimeSeries;

/*
 * Experiments for associative memory for a time-series of human evoluation
 * @author osogami
 */
public class EvolutionExperiment {

    public static void main(String[] args) {
	
	long startTimeMillis = System.currentTimeMillis();
	
	//
	// parsing arguments
	//
	int nNegative = Integer.parseInt(args[0]);
	int nPositive = Integer.parseInt(args[1]);
	int maxDelay = Integer.parseInt(args[2]);
	Parameter param = new Parameter(nNegative,nPositive);
	param.setMaxDelay(maxDelay);
	
	String directory = "Results/Evolution/";
	
	//
	// prepare training data
	//
	BitMapImage image = new BitMapImage();
	final TimeSeries trainingData = image.getEvolution();
	System.out.println("Training with");
	System.out.println(trainingData);
	
	//
	// prepare neural network
	//
	int N = trainingData.getDimension();	
	Network ann = new Network(N,1.0,param);
	
	try {
	    String filename = directory + "NN0param"+nPositive+"_"+nNegative+"delay"+maxDelay+".bin";
	    FileOutputStream file = new FileOutputStream(filename);
	    ObjectOutputStream os = new ObjectOutputStream(file);
	    os.writeObject(ann);
	    os.close();
	} catch (IOException e) {
	    e.printStackTrace();
	}
	
	//
	// training
	//
	int maxRepeat = 100;
	for(int n=1;n<=maxRepeat+1;n++){
	    
	    ann.train(trainingData,trainingData.size());//*minSize);
	    
	    if(n%(maxRepeat/10)==0){
		try {
		    String filename = directory + "NN"+n+"param"+nPositive+"_"+nNegative+"delay"+maxDelay+".bin";
		    FileOutputStream file = new FileOutputStream(filename);
		    ObjectOutputStream os = new ObjectOutputStream(file);
		    os.writeObject(ann);
		    os.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}
			
		// check completion
		ann.store();
		TimeSeries prediction = ann.freeRun(trainingData.size(),0);
		System.out.println("Iteration:"+n);
		System.out.println("prediction:");
		System.out.println(prediction);
		ann.restore();
		if(prediction.equals(trainingData)){
		    break;
		}else if(n==maxRepeat){
		    maxRepeat *= 10;
		}
	    }
	}

	long endTimeMillis = System.currentTimeMillis();
	long exeTimeSeconds = (endTimeMillis - startTimeMillis) / 1000;
	System.out.println("Completed in "+exeTimeSeconds+" seconds");		
    }
}

