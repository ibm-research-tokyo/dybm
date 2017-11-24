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
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;

/**
 * Dynamic Boltzmann machine
 * @author osogami
 *
 */
public class Network implements Serializable {

    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger("com.ibm.stdp.Network");
    
    Neuron[] neuron;
    List<Axon> axons;
    
    Parameter param;
    
    /**
     * Construct a network with n neurons
     * @param n
     */
    public Network(int n, Parameter parameter){
	param = parameter;
	
	NormalDistribution dist = new NormalDistribution(Parameter.Bias.mean,Parameter.Bias.std);
	dist.reseedRandomGenerator(0);
	double[] bias = dist.sample(n);
	
	neuron = new Neuron[n];
	for(int m=0;m<n;m++){
	    neuron[m] = new Neuron(param);
	    neuron[m].setBias(bias[m]);
	}
	
	axons = new ArrayList<Axon>();
    }
    
    /**
     * Construct a network with n neurons and connect each pair of neurons with probability p
     * with a random weight
     * @param n
     * @param p
     */
    public Network(int n, double p, Parameter parameter) {
	this(n,parameter);
	makeRandomConnection(p);
	makeRandomWeight();
    }
    
    /**
     * Connect each pair of neurons with probability p with a random delay
     * @param p
     */
    public void makeRandomConnection(double p){
	logger.info("Making random connection with probability "+p);
	UniformIntegerDistribution distDelay = new UniformIntegerDistribution(param.getMinDelay(), param.getMaxDelay());
	distDelay.reseedRandomGenerator(0);
	for(int i=0;i<sizeNeurons();i++){
	    for(int j=0;j<sizeNeurons();j++){
		if(Math.random()>p){
		    continue;
		}
		int delay = distDelay.sample();
		Axon a = new Axon(neuron[i],neuron[j],delay,param);
		axons.add(a);
	    }
	}
    }

    /**
     * Connect each pair of neurons with probability p
     * @param p
     */
    public void makeRandomWeight(){
	logger.info("Making random weight");
	NormalDistribution distU = new NormalDistribution(Parameter.Weight.U.mean,Parameter.Weight.U.std);
	distU.reseedRandomGenerator(0);
	NormalDistribution distV = new NormalDistribution(Parameter.Weight.V.mean,Parameter.Weight.V.std);
	distV.reseedRandomGenerator(0);
	for(Axon a : axons){
	    if(param.nRatesPositive>0){
		double[] u = distU.sample(param.nRatesPositive);
		a.setWeightU(u);
	    }
	    if(param.nRatesNegative>0){
		double[] v = distV.sample(param.nRatesNegative);
		a.setWeightV(v);
	    }			
	}
    }

    /**
     * Return the number of neurons
     * @return
     */
    public int sizeNeurons(){
	return neuron.length;
    }
    
    /**
     * Return the number of axons
     * @return
     */
    public int sizeAxons(){
	return axons.size();
    }
    
    /**
     * Train the neural network using the given training data for a given length
     * @param ts
     * @param length
     */
    public void train(TimeSeries ts, int length, int repeat) {
	double bestLL = Double.NEGATIVE_INFINITY;
	// multi start training
	for(int i=0;i<repeat;i++){
	    
	    // reset training data
	    ts.setIndex(0);
	    
	    // reset neural network
	    makeRandomWeight();
	    init();
	    initAdaGrad();
	    
	    // training
	    train(ts,length);
	    
	    // store the best
	    double LL = aveLL(ts,0,length);
	    if(LL>=bestLL || bestLL==Double.NEGATIVE_INFINITY){
		bestLL = LL;
		for(int j=0;j<axons.size();j++){
		    axons.get(j).storeWeight();
		}
		for(int j=0;j<neuron.length;j++){
		    neuron[j].storeBias();
		}
	    }
	    System.out.println("Final LL:"+LL);
	}
	
	// restore the best
	for(int j=0;j<axons.size();j++){
	    axons.get(j).restoreWeight();
	}
	for(int j=0;j<neuron.length;j++){
	    neuron[j].restoreBias();
	}		
    }

    /**
     * Initialize the state of the network
     */
    public void init() {
	for(Neuron n : neuron){
	    n.init();
	}
	for(Axon a : axons){
	    a.init();
	}		
    }
	
    /**
     * Initialize the parameters for AdaGrad
     */
    private void initAdaGrad() {
	for(Neuron n : neuron){
	    n.initAdaGrad();
	}
	for(Axon a : axons){
	    a.initAdaGrad();
	}
    }
    
    public void train(TimeSeries ts, int length) {
	logger.fine("Training for length "+length);
	
	for(int i=0;i<length;i++){
	    BinaryPattern bp = ts.getPattern();   
	    train(bp);
	    setValues(bp);
	}
    }

    /**
     * Train the neural network using the given pattern (one step)
     * @param bp
     */
    public void train(BinaryPattern bp){
	for(int i=0;i<sizeNeurons();i++){
	    neuron[i].train(bp.getValue(i));
	}
    }
    
    /**
     * Reading time series of length i data without training
     * @param ts
     * @param length
     * @return log likelihood of each step
     */
    public double[] read(TimeSeries ts, int length) {
	double[] loglikelihood = new double[length];
	for(int i=0;i<length;i++){
	    BinaryPattern bp = ts.getPattern();
	    loglikelihood[i] = LL(bp);
	    setValues(bp);
	}		
	return loglikelihood;
    }

    /**
     * Free run of the network for a given length with a given temperature
     * @param length
     * @param temperature
     * @return
     */
    public TimeSeries freeRun(int length, double temperature) {
	TimeSeries ts = new TimeSeries(length,sizeNeurons());
	for(int i=0;i<length;i++){
	    BinaryPattern bp = freeRun(temperature);
	    ts.set(i, bp);
	}
	return ts;
    }
    
    /**
     * One step free run of the network with a given temperature
     * @param temperature
     * @return
     */
    private BinaryPattern freeRun(double temperature){
	BinaryPattern bp = getSamples(temperature);
	setValues(bp);
	return bp;
    }
    
    /**
     * Get sample values from neurons
     * @param temperature: deterministic if 0
     * @return
     */
    private BinaryPattern getSamples(double temperature){
	boolean[] output = new boolean[sizeNeurons()];
	for(int i=0;i<sizeNeurons();i++){
	    output[i] = neuron[i].getSample(temperature);
	}
	BinaryPattern bp = new BinaryPattern(output);
	
	return bp;		
    }
    
    /**
     * Set the values of neurons with the given binary pattern
     * No training
     * @param bp
     * @param learning
     */
    private void setValues(BinaryPattern bp){
	for(int i=0;i<sizeNeurons();i++){
	    neuron[i].setValue(bp.getValue(i));
	}
    }
    
    /**
     * Return the i-th neuron
     * @param i
     * @return
     */
    public Neuron getNeuron(int i){
	return neuron[i];
    }
    
    public String toString() {
	String str = sizeNeurons()+" neurons:\n";
	for(Neuron n : neuron){
	    str += n.toString()+"\n";
	}		
	str += sizeAxons()+" axons:\n";
	for(Axon a : axons){
	    str += a.toString()+"\n";
	}
	return str;
    }
    
    public String toPython() {
	String str = "bias=dict()\n";
	for(Neuron n : neuron){
	    str += n.toPython();
	    str += "\n";
	}
	str += "delay=dict()\n";
	str += "U=dict()\n";
	str += "V=dict()\n";
	for(Axon a : axons){
	    str += a.toPython();
	    str += "\n";
	}
	return str;
    }
    
    public String toCSV() {
	String str = "";
	for(Neuron n : neuron){
	    str += n.toCSV();
	    str += "\n";
	}
	for(Axon a : axons){
	    str += a.toCSV();
	    str += "\n";
	}
	return str;
    }
    
    public String toLaTeX() {
	String str = "";
	for(Neuron n : neuron){
	    str += n.toLaTeX();
	    str += "\n";
	}
	for(Axon a : axons){
	    str += a.toLaTeX();
	    str += "\n";
	}
	return str;
    }
    
    /**
     * log likelihood of the training data, averaged over iterations and neurons
     * Store the state of the network, 
     * and restore the original
     * after the log likelihood is computed
     * @param trainingData
     */
    public double aveLL(TimeSeries trainingData, int start, int length) {
	// store and initialize
	for(Neuron n : neuron){
	    n.store();
	    n.init();
	}
	for(Axon a : axons){
	    a.store();
	    a.init();
	}
	trainingData.store();
	
	// evaluate the log likelihood
	trainingData.setIndex(start);
	double LL = 0;
	for(int i=0;i<length;i++){
	    BinaryPattern bp = trainingData.getPattern();
	    double oneStepLL = aveLL(bp) / length;
	    LL += oneStepLL;
	    setValues(bp);
	}
	
	// restore
	for(Neuron n : neuron){
	    n.restore();
	}
	for(Axon a : axons){
	    a.restore();
	}
	trainingData.restore();
	
	return LL;
    }

    /**
     * log likelihood of the training data
     * store the state of the network, 
     * and restore the original
     * after the log likelihood is computed
     * @param trainingData
     */
    public double LL(TimeSeries trainingData, int start, int length) {
	// store and initialize
	for(Neuron n : neuron){
	    n.store();
	    n.init();
	}
	for(Axon a : axons){
	    a.store();
	    a.init();
	}
	trainingData.store();
	
	// evaluate the log likelihood
	trainingData.setIndex(start);
	double LL = 0;
	for(int i=0;i<length;i++){
	    BinaryPattern bp = trainingData.getPattern();
	    double oneStepLL = LL(bp);
	    LL += oneStepLL;
	    setValues(bp);
	}
	
	// restore
	for(Neuron n : neuron){
	    n.restore();
	}
	for(Axon a : axons){
	    a.restore();
	}
	trainingData.restore();
	
	return LL;
    }
    
    private double aveLL(BinaryPattern bp){
	double LL = 0;
	for(int i=0;i<neuron.length;i++){
	    if(bp.getValue(i)==true){
		LL += neuron[i].getSpikeLL() / neuron.length;
	    }else{
		LL += neuron[i].getNonSpikeLL() / neuron.length;
	    }
	}
	return LL;
    }
    
    private double LL(BinaryPattern bp){
	double LL = 0;
	for(int i=0;i<neuron.length;i++){
	    if(bp.getValue(i)==true){
		LL += neuron[i].getSpikeLL();
	    }else{
		LL += neuron[i].getNonSpikeLL();
	    }
	}
	return LL;
    }
    
    public void store() {
	for(Neuron n : neuron){
	    n.store();
	}
	for(Axon a : axons){
	    a.store();
	}
    }
    
    public void restore() {
	for(Neuron n : neuron){
	    n.restore();
	}
	for(Axon a : axons){
	    a.restore();
	}
    }
}
