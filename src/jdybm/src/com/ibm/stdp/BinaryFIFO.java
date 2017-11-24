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


package com.ibm.stdp;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.logging.Logger;

/*
 * FIFO queue with binary representation
 * @author osogami
 */
public class BinaryFIFO implements FIFO, Serializable {

    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger("com.ibm.stdp.BinaryFIFO");
    
    /**
     * The length of the FIFO queue
     */
    private int length;
    
    /**
     * The FIFO queue
     */
    private LinkedList<Boolean> queue;
    
    /**
     * The space to store the backup of the FIFO queue
     */
    private LinkedList<Boolean> queueBackup;
    
    /**
     * Computed value of beta eligibility traces
     * This value is reset when the queue is updated
     */
    private double[] beta;
    
    /**
     * Pre-computed values that are used to compute beta
     */
    private double[][] muPower;
    
    private double[] rate;
    
    /**
     * Constracter of the FIFO queue with length n
     * @param n
     */
    public BinaryFIFO(int m, double[] mu) {
	length = m;
	rate = mu;
	beta = new double[rate.length];
	setMuPower();
	init();
    }
    
    @Override
    public void init() {
	queue = new LinkedList<Boolean>();
	for(int i=0;i<length;i++){
	    queue.offer(false);
	}
	
	resetBeta();
    }
    
    @Override
    public boolean offer(boolean x) {
	if(length==0 && CONNECT_ZERO){
	    return x;
	}
	queue.offer(x);
	boolean y = queue.poll();
	resetBeta();
	return y;
    }
    
    /**
     * Reset the value of the computed beta
     * Call this when the queue is updated
     */
    private void resetBeta(){
	for(int k=0;k<beta.length;k++){
	    beta[k] = Double.NaN;
	}		
    }
    
    @Override
    public double getBeta(int k) {
	if(Double.isNaN(beta[k])) {
	    double newBeta = 0;
	    for(int i=0;i<length;i++) {
		if(queue.get(i)) {
		    newBeta += muPower[k][i];
		}
	    }
	    beta[k] = newBeta;
	}
	
	return beta[k];
    }
    
    @SuppressWarnings("unused")
    private double getBetaFromScratch(int k) {
	// computing from scratch (becomes bottleneck)
	double mu = rate[k];
	double newBeta = 0;
	double pow = 1/mu; // mu**t
	for(int i=0;i<queue.size();i++){
	    if(queue.get(i)){
		//int t = -(i+1);
		//newBeta += Math.pow(mu, t);
		newBeta += pow;
	    }
	    pow /= mu;
	}
	return newBeta;
    }
    
    private void setMuPower(){
	muPower = new double[rate.length][length];
	for(int k=0;k<rate.length;k++){
	    double mu = rate[k];
	    for(int i=0;i<length;i++){
		int t = -(i+1);
		muPower[k][i] += Math.pow(mu, t);				
	    }
	}
    }
    
    @Override
    public void store() {
	queueBackup = new LinkedList<Boolean>();
	for(int i=0;i<length;i++){
	    queueBackup.offer(queue.get(i));
	}
    }
    
    @Override
    public void restore() {
	queue = queueBackup;
    }
    
    @Override
    public String toString() {
	String str = "";
	for(int i=length-1;i>=0;i--){
	    str += Boolean.toString(queue.get(i)).charAt(0)+",";
	}
	return str;
    }
}
