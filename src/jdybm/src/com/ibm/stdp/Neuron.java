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
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/*
 * Neuron of DyBM
 * @author osogami
 */
public class Neuron implements Serializable {

    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger("com.ibm.stdp.Neuron");
    
    /**
     * bias parameter of the neuron
     */
    private double bias;
    private double biasBackup;
    
    /**
     * list of axons of this neuron
     */
    private List<Axon> outgoingAxons;
    
    /**
     * list of incoming axons
     */
    private List<Axon> incomingAxons;
    
    /**
     * array of eligibility traces
     */
    private double[] eligibility;
    
    private double[] eligibilityBackup;
    
    /**
     * array of decay rates of eligibility traces
     */
    private double[] rate;
    
    /**
     * sum of squared gradient to be used for AdaGrad
     */
    private double totalSqGrad;
    
    /**
     * serial number of the neuron
     */
    int serialNumber;
    
    /**
     * serial number of neurons
     */
    static int serial = 0;
    
    // only for debugging purposes to store the historical values
    public List<Boolean> historyValue;
    public List<Double> historyBias;
    public List<Double> historyEligibility;
    
    public Neuron(Parameter param) {
	serialNumber = serial++;
	
	rate = param.rateNegative;
	eligibility = new double[param.nRatesNegative];
	outgoingAxons = new ArrayList<Axon>();
	incomingAxons = new ArrayList<Axon>();
	totalSqGrad = 0;
	
	init();
	
	// only for debugging purposes
	historyValue = new ArrayList<Boolean>();
	historyBias = new ArrayList<Double>();
	historyEligibility = new ArrayList<Double>();
    }
    
    /**
     * Set the value of the bias parameter of the neuron
     * @param b
     */
    public void setBias(double b){
	if(Double.isNaN(b)){
	    logger.warning("setting NaN bias");
	}
	bias = b;
    }
    
    /**
     * Set the value of the neuron and update the eligibility trace
     * @param x
     */
    public void setValue(boolean x){
	updateEligibility(x);
	
	// propagate the previous value to the axons
	propagate(x);
	
	if(Parameter.keepHistory){
	    historyValue.add(x);
	}
    }
    
    /**
     * Update the value of the eligibility trace with value
     * @param value
     */
    private void updateEligibility(boolean x){
	for(int i=0;i<eligibility.length;i++){
	    if(x){
		eligibility[i] = rate[i] * (eligibility[i] + 1);
	    }else{
		eligibility[i] = rate[i] * eligibility[i];
	    }
	}
	
	if(Parameter.keepHistory && eligibility.length>0){
	    historyEligibility.add(eligibility[0]);
	}
    }
    
    /**
     * Update the bias and weight parameters
     * after seeing the value x
     * @param x
     */
    public void train(boolean x){
	double expected = getExpectation();
	
	updateBias(x,expected);
	
	for(Axon a : incomingAxons){
	    a.updateWeightFromPost(x,expected);
	}
	
	for(Axon a : outgoingAxons){
	    a.updateWeightFromPre(x, expected);
	}
    }
    
    /**
     * Update the bias of the neuron
     * after seeing the value x
     * @param x
     */
    private void updateBias(boolean x, double expected){
	double g;
	if(x){
	    //bias += Parameter.learningRate * (1-expected);
	    g = 1 - expected;
	}else{
	    //bias -= Parameter.learningRate * expected;
	    g = - expected;
	}
	if(g==0){
	    return;
	}
	totalSqGrad += g*g;
	if(totalSqGrad==0){
	    // this can happen when g is small that g>0 but g*g=0
	    bias += Parameter.learningRate;			
	}else{
	    bias += Parameter.learningRate / Math.sqrt(totalSqGrad) * g;
	}
	
	if(Double.isNaN(bias)){
	    logger.warning(
			   "Updating bias to NaN: "
			   + "delta="+Parameter.learningRate+"/sqrt("+totalSqGrad+")*"+g);
	}
	
	if(Parameter.keepHistory){
	    historyBias.add(bias);
	}
    }
    
    /**
     * Propagate the current value to axons
     */
    private void propagate(boolean x){
	for(Axon a : outgoingAxons){
	    a.offer(x);
	}
    }
    
    /**
     * Generate a sample, according to the spiking probability
     * @return
     */
    public boolean getSample(double temperature){
	double p = getSpikeProbability(temperature);
	if(p>=1){
	    return true;
	}else if(p<=0){
	    return false;
	}
	double u = Math.random();
	if(u<p){
	    return true;
	}else{
	    return false;
	}
    }
    
    /**
     * Return the expected value of the neuron in the next moment
     * @return
     */
    public double getExpectation(){
	return getSpikeProbability();
    }
    
    /**
     * Return the log likelihood of the neuron going to spike in the next moment
     * @return e - log(1+exp(e)),
     * where e = -energy/temperature
     */
    public double getSpikeLL() {
	double e = - getSpikeEnergy() / Parameter.temperature;
	double LL;
	if(Math.exp(e)==Double.POSITIVE_INFINITY){
	    // x - log(1+exp(x)) = - log(1+exp(-x))
	    LL = - Math.log(1+Math.exp(-e));
	}else{
	    LL = e - Math.log(1+Math.exp(e));
	}
	
	if(Double.isNaN(LL)){
	    logger.warning("getSpikeLL returning NaN");
	}			
	
	return LL;
    }
    
    public double getNonSpikeLL() {
	double e = - getSpikeEnergy() / Parameter.temperature;
	double LL;
	if(Math.exp(e)==Double.POSITIVE_INFINITY){
	    // - log(1+exp(x)) = - x - log(1+exp(-x))
	    LL = - e - Math.log(1+Math.exp(-e));
	}else{
	    LL = - Math.log(1+Math.exp(e));
	}
	
	if(Double.isNaN(LL)){
	    logger.warning("getNonSpikeLL returning NaN");
	}			
	
	return LL;
    }
    
    /**
     * Return the probability that the neuron is going to spike in the next moment
     * @return probability = exp(-E/tau) / (1 + exp(-E/tau)),
     * where E = energy given a spike
     *       tau = temperature
     */
    public double getSpikeProbability(){
	double e = Math.exp(-getSpikeEnergy()/Parameter.temperature);
	if(e==Double.POSITIVE_INFINITY){
	    return 1;
	}
	double p = e/(1+e);
	return p;
    }
    
    /**
     * Return the probability that the neuron is going to spike in the next moment
     * @return probability = exp(-E/tau) / (1 + exp(-E/tau)),
     * where E = energy given a spike
     *       tau = temperature
     */
    public double getSpikeProbability(double temperature){
	if(temperature==0){
	    double energy = getSpikeEnergy();
	    if(energy<0){
		return 1;
	    }else if(energy>0){
		return 0;
	    }else{
		return 0.5;
	    }
	}else{
	    double e = Math.exp(-getSpikeEnergy()/temperature);
	    return e/(1+e);
	}
    }
    
    /**
     * Return the energy of the neuron,
     * assuming that it will spike in the next moment
     * 
     * @return
     */
    private double getSpikeEnergy(){
	// energy term that depends only on this neuron
	// - b[j] x[j]
	double energy = - getBias();
	
	for(Axon a : getIncomingAxons()){
	    // energy term that depends on an outgoing axon
	    // - sum_k u[i,j,k] alpha[i,j,k] + sum_k v[i,j,k] beta[i,j,k]
	    energy += a.getPostSpikeEnergy();
	}
	
	for(Axon a : getIncomingAxons()){
	    // energy term that depends on an incoming axon
	    // sum_k v[j,i,k] gamma[i,k]
	    energy += a.getPreSpikeEnergy();
	}
	
	if(Double.isNaN(energy)){
	    logger.warning("getSpikeEnergy returning NaN");
	}			
	
	return energy;
    }
    
    /**
     * Add an action to the list of axons
     * @param a
     */
    public void addOutgoingAxon(Axon a){
	outgoingAxons.add(a);
    }
    
    /**
     * Add an action to the list of axons
     * @param a
     */
    public void addIncomingAxon(Axon a){
	incomingAxons.add(a);
    }
    
    /**
     * Return the value of the eligibility trace
     * @return
     */
    public double getEligibility(int k) {
	return eligibility[k];
    }
    
    public double[] getEligibility() {
	return eligibility;
    }
    
    /**
     * Return the bias of the neuron
     * @return
     */
    public double getBias(){
	return bias;
    }
    
    /**
     * Return the list of axons that originate from the neuron
     * @return
     */
    private List<Axon> getOutgoingAxons(){
	return outgoingAxons;
    }
    
    /**
     * Return the list of axons that come into the neuron
     * @return
     */
    private List<Axon> getIncomingAxons(){
	return incomingAxons;
    }
    
    /**
     * Store the current eligibility traces
     */
    public void store(){
	eligibilityBackup = eligibility.clone();
    }
    
    /**
     * Restore the stored eligibility traces
     */
    public void restore(){
	eligibility = eligibilityBackup;
    }
    
    /**
     * Initialize eligibility traces
     */
    public void init() {
	for(int i=0;i<eligibility.length;i++){
	    eligibility[i] = 0;
	}
    }
    
    /**
     * Store the value of the bias
     */
    public void storeBias(){
	biasBackup = bias;
    }
    
    /**
     * Restore the value of the bias
     */
    public void restoreBias(){
	bias = biasBackup;
    }
    
    public String toString() {
	// bias
	String b = Double.toString(getBias());
	b = (String) b.subSequence(0, Math.min(10, b.length()));
	
	// probability
	double prob = getSpikeProbability();
	String p;
	if(prob<0.01){
	    p = "0.00";
	}else{
	    p = Double.toString(prob);
	    p = (String) p.subSequence(0, Math.min(4, p.length()));
	}
	
	// eligibility traces
	double[] e = new double[eligibility.length];
	for(int k=0;k<eligibility.length;k++){
	    e[k] = getEligibility(k);
	}
	
	String str = "N"+serialNumber+":\t"
	    + "B:"+b+",\t"
	    + "prob:"+p+",\t"
	    + "elig:"+Arrays.toString(e);
	return str;
    }
    
    public String toPython() {
	String b = Double.toString(getBias());
	String str = "bias["+serialNumber+"]="+b;
	return str;
    }	
    
    public String toCSV() {
	NumberFormat format = NumberFormat.getInstance();
	format.setMaximumFractionDigits(9);
	format.setMinimumFractionDigits(9);
	String str = "bias,"+serialNumber+","+format.format(getBias());
	return str;
    }	
    
    public String toLaTeX() {
	NumberFormat format = NumberFormat.getInstance();
	format.setMaximumFractionDigits(9);
	format.setMinimumFractionDigits(9);
	int i = serialNumber + 1;
	String str = "bias,"+i+" & & "+format.format(getBias()) + "\\\\";
	return str;
    }	
    
    /**
     * Initialize the parameter for AdaGrad
     */
    public void initAdaGrad() {
	totalSqGrad = 0.0;
    }
}
