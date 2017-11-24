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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import com.ibm.stdp.Network;
import com.ibm.stdp.Parameter;
import com.ibm.stdp.TimeSeries;

/*
 * Experiments for associative memory for two time-series, forward and backward "SCIENCE"
 * @author osogami
 */
public class MirrorExperiment3 {
	
    static int timeTest;
    static int timeTrain;
    
    public static void main(String[] args) {
	
	long startTimeMillis = System.currentTimeMillis();
	
	//
	// parsing arguments
	//
	String data = args[0];
	int cueLength = Integer.parseInt(args[1]);
	int nNegative = Integer.parseInt(args[2]);
	int nPositive = Integer.parseInt(args[3]);
	int maxDelay = Integer.parseInt(args[4]);
	Parameter param = new Parameter(nNegative,nPositive);
	param.setMaxDelay(maxDelay);
	
	String directory = "Results/"+data+"/Mirror3b/";
	String midResult;// = directory+"NN_cue"+cueLength+"param"+nNegative+"_"+nPositive+".bin";
	boolean resume = false;
	if(args.length>5){
	    resume = true;
	}
	
	//
	// prepare training data
	//
	final TimeSeries trainingData = new TimeSeries(data);
	final TimeSeries mirrorData = trainingData.mirror();
	
	System.out.println("Training with");
	System.out.println(trainingData);
	System.out.println("followed by");
	System.out.println(mirrorData);
	
	//
	// prepare validation set
	//
	String cueString = data.substring(0, cueLength);	// from 0 to cueLength - 1
	final TimeSeries cueData = new TimeSeries(cueString);
	
	String ansString = data.substring(cueLength) + cueString;
	//String ansString = data.substring(cueLength);		// from cueLength to the end
	final TimeSeries ansData = new TimeSeries(ansString);
	
	System.out.println("Forward cue (length "+cueLength+"):");
	System.out.println(cueData);
	System.out.println("To predict");
	System.out.println(ansData);
	
	cueString = data.substring(data.length()-cueLength);	// from data.length() - cueLength to the end
	TimeSeries tmpData = new TimeSeries(cueString);
	final TimeSeries cueMirror = tmpData.mirror();
	
	ansString = cueString + data.substring(0,data.length()-cueLength);
	//ansString = data.substring(0,data.length()-cueLength);	// from 0 to data.length - cueLength - 1
	tmpData = new TimeSeries(ansString);
	final TimeSeries ansMirror = tmpData.mirror();
	
	System.out.println("Backward cue (length "+cueLength+"):");
	System.out.println(cueMirror);
	System.out.println("To predict");
	System.out.println(ansMirror);
	
	//
	// prepare neural network
	//
	Network ann;
	
	if(resume){
	    try {
		midResult = args[5];
		FileInputStream fis = new FileInputStream(midResult);
		ObjectInputStream is = new ObjectInputStream(fis);
		ann = (Network) is.readObject();
		is.close();
	    } catch (IOException | ClassNotFoundException e) {
		e.printStackTrace();
		return;
	    }			
	}else{
	    int N = trainingData.getDimension();
	    ann = new Network(N,1.0,param);
	}
	
	timeTest = 0;
	timeTrain = 0;
	
	//
	// training
	//
	int maxIteration = Integer.MAX_VALUE;
	int writeStep = 1;
	List<Integer> forwardIteration = new ArrayList<Integer>();
	List<Integer> backwardIteration = new ArrayList<Integer>();
	int n = 0;
	while(true){
	    
	    // store intermediate results
	    if(n%writeStep==0){
		try {
		    String filename = directory+"NN"+n+"_cue"+cueLength+"param"+nNegative+"_"+nPositive+"delay"+maxDelay+".bin";
		    FileOutputStream fos = new FileOutputStream(filename);
		    ObjectOutputStream os = new ObjectOutputStream(fos);
		    os.writeObject(ann);
		    os.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}
	    }
	    if(n==writeStep*10){
		writeStep *= 10;
	    }
	    
	    // to reset eligibility traces before resuming forward training
	    ann.read(trainingData,trainingData.size());
	    
	    // forward training
	    int i;
	    for(i=0;i<maxIteration;i++){
		double t0 = System.currentTimeMillis();
		
		// test if already trained
		ann.store();
		ann.init();
		ann.read(cueData,cueData.size());				
		TimeSeries prediction = ann.freeRun(ansData.getPeriod(),0);
		ann.restore();
		
		if(prediction.equals(ansData)){
		    System.out.println("Trained forward for "+i+" iteration");
		    forwardIteration.add(i);
		    break;
		}
		
		double t1 = System.currentTimeMillis();
		
		// train
		ann.train(trainingData,trainingData.size());
		
		double t2 = System.currentTimeMillis();
		timeTest += t1-t0;
		timeTrain += t2-t1;
	    }
	    if(i==0 && n>0){
		System.out.println("Successfully completed");
		break;
	    }
	    if(i==maxIteration){
		System.out.println("Forward training reached max iteration:"+maxIteration);
		maxIteration *= 2;
	    }
	    
	    n++;
	    
	    // store intermediate results
	    if(n==1 || n==3){
		try {
		    String filename = directory+"NN"+n+"_cue"+cueLength+"param"+nNegative+"_"+nPositive+"delay"+maxDelay+".bin";
		    FileOutputStream fos = new FileOutputStream(filename);
		    ObjectOutputStream os = new ObjectOutputStream(fos);
		    os.writeObject(ann);
		    os.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}
	    }
	    
	    //
	    // backward training
	    //
	    
	    // to reset eligibility traces before resuming backward training
	    ann.read(mirrorData,mirrorData.size());
	    
	    for(i=0;i<maxIteration;i++){
		double t0 = System.currentTimeMillis();
		
		// test if already trained
		ann.store();
		ann.init();
		ann.read(cueMirror,cueMirror.size());
		TimeSeries prediction = ann.freeRun(ansMirror.getPeriod(),0);
		ann.restore();
		
		if(prediction.equals(ansMirror)){
		    System.out.println("Trained backward for "+i+" iteration");
		    backwardIteration.add(i);
		    break;
		}
		
		double t1 = System.currentTimeMillis();
		
		// train
		ann.train(mirrorData,mirrorData.size());
		
		double t2 = System.currentTimeMillis();
		timeTest += t1-t0;
		timeTrain += t2-t1;
	    }
	    if(i==0){
		System.out.println("Successfully completed");
		break;
	    }
	    if(i==maxIteration){
		System.out.println("Backward training reached max iteration:"+maxIteration);
		maxIteration *= 2;
	    }
	    
	    n++;
	    
	    System.out.println("Time to test: "+timeTest);
	    System.out.println("Time to train: "+timeTrain);
	}
	
	//
	// store final results
	//
	try {
	    String filename = directory+"NN"+n+"_cue"+cueLength+"param"+nNegative+"_"+nPositive+"delay"+maxDelay+".bin";
	    FileOutputStream fos = new FileOutputStream(filename);
	    ObjectOutputStream os = new ObjectOutputStream(fos);
	    os.writeObject(ann);
	    os.close();
	} catch (IOException e) {
	    e.printStackTrace();
	}
	
	try {
	    String filename = directory+"Forward"+n+"_cue"+cueLength+"param"+nNegative+"_"+nPositive+"delay"+maxDelay+".csv";
	    File f = new File(filename);
	    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(f)));
	    for(Integer num : forwardIteration){
		pw.println(num);
	    }
	    pw.close();
	} catch (IOException e) {					
	    e.printStackTrace();
	}
	
	try {
	    String filename = directory+"Backward"+n+"_cue"+cueLength+"param"+nNegative+"_"+nPositive+"delay"+maxDelay+".csv";
	    File f = new File(filename);
	    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(f)));
	    for(Integer num : backwardIteration){
		pw.println(num);
	    }
	    pw.close();
	} catch (IOException e) {					
	    e.printStackTrace();
	}
		
	ann.store();
	
	// reading cue (without training)
	System.out.println("Giving forward cue:");
	System.out.println(cueData);
	ann.init();
	ann.read(cueData,cueData.size());
	
	// prediction
	TimeSeries prediction = ann.freeRun(trainingData.getPeriod(),0);		
	System.out.println("\nPrediction:");
	System.out.println(prediction);
	
	// reading cue (without training)
	System.out.println("Giving backward cue:");
	System.out.println(cueMirror);
	ann.init();
	ann.read(cueMirror,cueMirror.size());
	
	// prediction
	TimeSeries prediction2 = ann.freeRun(mirrorData.getPeriod(),0);		
	System.out.println("\nPrediction:");
	System.out.println(prediction2);
	
	ann.restore();
	
	long endTimeMillis = System.currentTimeMillis();
	long exeTimeSeconds = (endTimeMillis - startTimeMillis) / 1000;
	System.out.println("Completed in "+exeTimeSeconds+" seconds");		
    }	
}
