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
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

/*
 * Conection in DyBM
 * @author osogami
 */
public class Axon implements Serializable {

    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger("com.ibm.stdp.Axon");
    
    /**
     * pre synaptic neuron
     */
    private Neuron pre;
    
    /**
     * post synaptic neuron
     */
    private Neuron post;
    
    /**
     * synaptic delay
     */
    private int delay;
    
    /**
     * rate of decay for eligibility traces
     */
    private double[] rate;
    
    /**
     * FIFO queue
     */
    private FIFO fifo;
    
    /**
     * eligibility traces
     */
    private double[] eligibility;
    private double[] eligibilityBackup;
    
    /**
     * sum of squared gradient to be used for AdaGrad
     */
    private double[] totalSqGradU;
    private double[] totalSqGradV;
    
    /**
     * weight parameters
     */
    private double[] weightU;
    private double[] weightV;
    private double[] weightUBackup;
    private double[] weightVBackup;
    
    // only for debugging purposes
    public List<Boolean> history;
    
    /**
     * Axon from pre-synaptic neuron, n0, to post-synaptic neuron, n1, with delay, d
     * @param n0
     * @param n1
     * @param d
     */
    public Axon(Neuron n0, Neuron n1, int d, Parameter param){
	setNeurons(n0,n1);
	
	delay = d;
	rate = param.ratePositive;
	
	weightU = new double[param.nRatesPositive];
	weightV = new double[param.nRatesNegative];
	eligibility = new double[param.nRatesPositive];
	
	fifo = new IntFIFO(delay-1,param.rateNegative);
	//fifo = new BinaryFIFO(delay-1);
	
	init();		
	initAdaGrad();
	
	// only for debugging purposes
	history = new ArrayList<Boolean>();
    }
    
    /**
     * Initialize the FIFO queue and eligibility traces
     */
    public void init() {
	// Initialize FIFO queue
	fifo.init();
	
	// Initialize eligibility trace
	for(int i=0;i<eligibility.length;i++){
	    eligibility[i] = 0;
	}
    }
    
    /**
     * Initialize the parameters for AdaGrad
     */
    public void initAdaGrad() {
	totalSqGradU = new double[weightU.length];
	totalSqGradV = new double[weightV.length];		
    }
    
    /**
     * Get the value of the k-th eligibility trace associated with the FIFO queue
     * @param k
     * @return
     */
    private double getBeta(int k){
	return fifo.getBeta(k);
    }
    
    /**
     * Set pre synaptic neuron, n0, and post synaptic neuron, n1
     * @param n0
     * @param n1
     */
    public void setNeurons(Neuron n0, Neuron n1){
	pre = n0;
	post = n1;
	n0.addOutgoingAxon(this);
	n1.addIncomingAxon(this);		
    }
    
    /** 
     * Return the energy associate with the incoming axon,
     * assuming that the post-synaptic neuron spikes
     * @return
     */
    public double getPostSpikeEnergy() {
	double energy = 0;
	// - sum_k u[i,j,k] alpha[i,j,k]
	for(int k=0;k<weightU.length;k++){
	    if(eligibility[k]==0){
		// weightU[k] might be infinity
		continue;
	    }
	    energy -= weightU[k] * eligibility[k];
	}
	// + sum_k v[i,j,k] beta[i,j,k]
	for(int k=0;k<weightV.length;k++){
	    double beta = getBeta(k);
	    if(beta==0){
		// weightV[k] might be infinity
		continue;
	    }
	    energy += weightV[k] * getBeta(k);
	}
	
	if(Double.isNaN(energy)){
	    logger.warning("gestPostSpikeEnergy returning NaN");
	}
	
	return energy;
    }
    
    /** 
     * Return the energy associate with the outgoing axon,
     * assuming that the pre-synaptic neuron spikes
     * @return
     */
    public double getPreSpikeEnergy() {
	double energy = 0;
	// sum_k v[j,i,k] gamma[i,k]
	for(int k=0;k<weightV.length;k++){
	    double e = post.getEligibility(k);
	    if(e==0){
		// weightV[k] might be infinity
		continue;
	    }
	    energy += weightV[k] * e;
	}
	
	if(Double.isNaN(energy)){
	    logger.warning("gestPreSpikeEnergy returning NaN");
	}
	
	return energy;
    }
    
    /**
     * Add a new value to (the tail of) the axon
     * @param value
     */
    public void offer(boolean x) {
	boolean y = fifo.offer(x);
	
	updateEligibility(y);
	
	// only for debugging purposes
	if(Parameter.keepHistory){
	    history.add(x);
	}
    }
    
    /**
     * Update the synaptic weight based on the value of the post synaptic neuron
     * @param x
     * @param expected
     */
    public void updateWeightFromPost(boolean x, double expected) {
	// update u
	for(int k=0;k<weightU.length;k++){
	    double g;
	    if(x){
		g = eligibility[k] * (1-expected);
	    }else{
		g = -eligibility[k] * expected;
	    }
	    if(g==0){
		continue;
	    }
	    totalSqGradU[k] += g*g;
	    if(totalSqGradU[k]==0){
		// this can happen when g is small that g>0 but g*g=0
		weightU[k] += Parameter.learningRate;
	    }else{
		weightU[k] += Parameter.learningRate / Math.sqrt(totalSqGradU[k]) * g;
	    }
	    
	    if(Double.isInfinite(weightU[k]) || Double.isNaN(weightU[k])){
		logger.warning(
			       "updateWeightFromPost updated weightU["+k+"] to "+weightU[k]
			       +" by "+Parameter.learningRate+"/("+totalSqGradU[k]+")*"+g);
	    }
	}
	
	// update v
	for(int k=0;k<weightV.length;k++){
	    double g;
	    if(x){
		g = -getBeta(k) * (1-expected);
	    }else{
		g = getBeta(k) * expected;
	    }
	    if(g==0){
		continue;
	    }
	    totalSqGradV[k] += g*g;
	    if(totalSqGradV[k]==0){
		// this can happen when g is small that g>0 but g*g=0
		weightV[k] += Parameter.learningRate;				
	    }else{
		weightV[k] += Parameter.learningRate / Math.sqrt(totalSqGradV[k]) * g;
	    }
	    
	    if(Double.isInfinite(weightV[k]) || Double.isNaN(weightV[k])){
		logger.warning(
			       "updateWeightFromPost updated weightV["+k+"] to "+weightV[k]
			       +" by "+Parameter.learningRate+"/sqrt("+totalSqGradV[k]+")*"+g);
	    }
	}
    }
    
    /**
     * Update the synaptic weight based on the value of the pre synaptic neuron
     * @param x
     * @param expected
     */
    public void updateWeightFromPre(boolean x, double expected) {
	for(int k=0;k<weightV.length;k++){
	    double g;
	    if(x){
		g = -post.getEligibility(k) * (1-expected);
	    }else{
		g = post.getEligibility(k) * expected;
	    }
	    if(g*g==0){
		// strictly speaking, this should be g==0
		// however, g*g==0 would cause division by 0 in the following step
		continue;
	    }
	    totalSqGradV[k] += g*g;
	    weightV[k] += Parameter.learningRate / Math.sqrt(totalSqGradV[k]) * g;
	    
	    if(Double.isInfinite(weightV[k]) || Double.isNaN(weightV[k])){
		logger.warning(
			       "updateWeightFromPre updated weightV["+k+"] to "+weightV[k]
			       +" by "+Parameter.learningRate+"/sqrt("+totalSqGradV[k]+")*"+g);
	    }
	}
    }
    
    /**
     * Update the value of the eligibility traces
     * @param y
     */
    private void updateEligibility(boolean y) {
	for(int k=0;k<rate.length;k++){
	    if(y){
		eligibility[k] = rate[k] * (eligibility[k] + 1);
	    }else{
		eligibility[k] = rate[k] * eligibility[k];
	    }
	}		
    }
    
    /**
     * Set the values of the weight, u
     * @param w
     */
    public void setWeightU(double[] w){
	for(int k=0;k<weightU.length;k++){
	    weightU[k] = w[k];
	}
    }
    
    /**
     * Set the values of the weight, v
     * @param w
     */
    public void setWeightV(double[] w){
	for(int k=0;k<weightV.length;k++){
	    weightV[k] = w[k];
	}
    }
    
    /**
     * Set the k-th value of the weight, u
     * @param k
     * @param w
     */
    public void setWeightU(int k, double w){
	weightU[k] = w;
    }
    
    /**
     * Set the k-th value of the weight, v
     * @param k
     * @param w
     */
    public void setWeightV(int k, double w){
	weightV[k] = w;
    }
    
    public double[] getWeightU() {
	return weightU;
    }
    
    public double[] getWeightV() {
	return weightV;
    }
    
    /**
     * Get pre synaptic neuron
     * @return
     */
    public Neuron getPreNeuron() {
	return pre;
    }
    
    /**
     * Get post synaptic neuron
     * @return
     */
    public Neuron getPostNeuron() {
	return post;
    }
    
    /**
     * Store the current FIFO queue and eligibility traces
     */
    public void store(){
	fifo.store();
	
	// Store eligibility trace
	eligibilityBackup = eligibility.clone();
    }
    
    /**
     * Restore the stored FIFO queue and eligibility traces
     */
    public void restore(){
	fifo.restore();
	
	// Restore eligibility trace
	eligibility = eligibilityBackup;	
    }
    
    /**
     * Store the values of the weight
     */
    public void storeWeight(){
	// store u
	weightUBackup = weightU.clone();
	
	// store v
	weightVBackup = weightV.clone();
    }
    
    /**
     * Restore the values of the weight
     */
    public void restoreWeight(){
	weightU = weightUBackup;
	weightV = weightVBackup;
    }
    
    public String toString() {
	String str = "N"+pre.serialNumber+"->N"+post.serialNumber+":\t";
	
	str += "fifo:" + fifo.toString() + "\t";
	
	str += "U:";
	for(double u: weightU){
	    if(Double.toString(u).length()<10){
		str += Double.toString(u)+",";
	    }else{
		str += Double.toString(u).subSequence(0, 10)+",";	
	    }
	}
	str += "\t";
	
	str += "V:";
	for(double v: weightV){
	    if(Double.toString(v).length()<10){
		str += Double.toString(v)+",";
	    }else{
		str += Double.toString(v).subSequence(0, 10)+",";
	    }
	}
	str += "\t";
	
	str += "elig:"+Arrays.toString(eligibility);
	
	return str;
    }
    
    public String toPython() {
	String str = "delay[("+pre.serialNumber+","+post.serialNumber+")]=";
	str += Integer.toString(delay);
	str += "\n";
	str += "U[("+pre.serialNumber+","+post.serialNumber+")]=";
	str += Arrays.toString(weightU);
	str += "\n";
	str += "V[("+pre.serialNumber+","+post.serialNumber+")]=";
	str += Arrays.toString(weightV);
	return str;
    }
    
    public String toCSV() {
	NumberFormat format = NumberFormat.getInstance();
	format.setMaximumFractionDigits(9);
	format.setMinimumFractionDigits(9);
	
	String str = "delay,"+pre.serialNumber+","+post.serialNumber+","+delay;
	str += "\n";
	str += "U,"+pre.serialNumber+","+post.serialNumber+",";
	for(int i=0;i<weightU.length;i++){
	    double u = weightU[i];
	    str += format.format(u);
	    if(i<weightU.length-1){
		str += ",";
	    }
	}
	str += "\n";
	str += "V,"+pre.serialNumber+","+post.serialNumber+",";
	for(int i=0;i<weightV.length;i++){
	    double v = weightV[i];
	    str += format.format(v);
	    if(i<weightV.length-1){
		str += ",";
	    }
	}
	return str;
    }
    
    public String toLaTeX() {
	NumberFormat format = NumberFormat.getInstance();
	format.setMaximumFractionDigits(9);
	format.setMinimumFractionDigits(9);
	
	int i = pre.serialNumber + 1;
	int j = post.serialNumber + 1;
	
	String str = "delay,"+i+" & & "+j+" & & "+delay+"\\\\";
	str += "\n";
	str += "U,"+i+" & & "+j+" & & ";
	for(int k=0;k<weightU.length;k++){
	    double u = weightU[k];
	    str += format.format(u);
	    if(k<weightU.length-1){
		str += " & & ";
	    }else{
		str += "\\\\";
	    }
	}
	str += "\n";
	str += "V,"+i+" & & "+j+" & & ";
	for(int k=0;k<weightV.length;k++){
	    double v = weightV[k];
	    str += format.format(v);
	    if(k<weightV.length-1){
		str += " & & ";
	    }else{
		str += "\\\\";
	    }
	}
	return str;
    }
}
