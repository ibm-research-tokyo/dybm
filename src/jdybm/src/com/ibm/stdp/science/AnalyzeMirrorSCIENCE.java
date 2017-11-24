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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import com.ibm.stdp.Network;
import com.ibm.stdp.TimeSeries;

/*
 * Analysis of the results of the Mirror experiments
 * @author osogami
 */
public class AnalyzeMirrorSCIENCE {
	
    public static void main(String[] args) {
	final String data = "SCIENCE";
	int cueLength = 5;
	String directory = "Results/SCIENCE/Mirror3b/";
	
	final TimeSeries trainingData = new TimeSeries(data);
	
	String cueString = data.substring(0, cueLength);	// from 0 to cueLength - 1
	final TimeSeries cueData = new TimeSeries(cueString);
	
	//String ansString = data.substring(cueLength) + cueString;
	String ansString = data.substring(cueLength);		// from cueLength to the end
	final TimeSeries ansData = new TimeSeries(ansString);
	
	cueString = data.substring(data.length()-cueLength);	// from data.length() - cueLength to the end
	TimeSeries tmpData = new TimeSeries(cueString);
	final TimeSeries cueMirror = tmpData.mirror();
	
	//ansString = cueString + data.substring(0,data.length()-cueLength);
	ansString = data.substring(0,data.length()-cueLength);	// from 0 to data.length - cueLength - 1
	TimeSeries tmpData2 = new TimeSeries(ansString);
	final TimeSeries ansMirror = tmpData2.mirror();
	
	System.out.println("x = dict()");
	System.out.println("y = dict()");
	
	File dir = new File(directory);
	File[] files = dir.listFiles();
	for(File file : files){
	    String filename = file.getName();
	    if(!filename.endsWith(".bin")){
		continue;
	    }
	    String number = filename.split("_")[0].split("NN")[1];
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
	    
	    ann.init();
	    ann.read(cueData, cueData.size());
	    
	    // prediction
	    TimeSeries prediction = ann.freeRun(ansData.size()+cueData.size(),0);
	    System.out.println("x["+n+"] = "+prediction.toPython());
	    
	    ann.init();
	    ann.read(cueMirror, cueMirror.size());
	    
	    // prediction
	    TimeSeries prediction2 = ann.freeRun(ansMirror.size()+cueMirror.size(),0);		
	    System.out.println("y["+n+"] = "+prediction2.toPython());
	}
	
	System.out.println("forwardCue = "+cueData.toPython());
	System.out.println("backwardCue = "+cueMirror.toPython());
    }
}
