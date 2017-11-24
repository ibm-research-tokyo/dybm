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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import com.ibm.stdp.Network;
import com.ibm.stdp.Parameter;
import com.ibm.stdp.TimeSeries;

/*
 * Experiments for associative memory for a single time-series "SCIENCE"
 * @author osogami
 */
public class SingleExperiment3 {

    public static void main(String[] args) {
	long startTimeMillis = System.currentTimeMillis();
	
	//
	// parsing arguments
	//
	String data = args[0];
	int nNegative = Integer.parseInt(args[1]);
	int nPositive = Integer.parseInt(args[2]);
	int maxDelay = Integer.parseInt(args[3]);
	Parameter param = new Parameter(nNegative,nPositive);
	param.setMaxDelay(maxDelay);
	
	String directory = "Results/"+data+"/Single3b/";
	
	//
	// prepare training data
	//
	final TimeSeries trainingData = new TimeSeries(data);
	System.out.println("Training with");
	System.out.println(trainingData);
	
	//
	// prepare neural network
	//
	int N = trainingData.getDimension();	
	Network ann = new Network(N,1.0,param);
	
	// dumping network before training
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
	for(int n=1;n<maxRepeat+1;n++){
	    
	    ann.train(trainingData,trainingData.size());//*minSize);
	    
	    if(n%(maxRepeat/10)==0){
		// dump network under training
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
