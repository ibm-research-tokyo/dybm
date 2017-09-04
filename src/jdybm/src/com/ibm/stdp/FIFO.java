// (C) Copyright IBM Corp. 2015

package com.ibm.stdp;

/*
 * FIFO queue
 * @author osogami
 */
public interface FIFO {

    public boolean CONNECT_ZERO = false;
    
    /**
     * Initialize the values of the FIFO queue
     */
    public void init();
    
    /**
     * Add a value, x, to the tail of the FIFO queue, and
     * remove the value from the head of the FIFO queue
     * @param x
     * @return the value removed from the head
     */
    public boolean offer(boolean x);
    
    /**
     * Get the value of the k-th beta eligibility trace of the FIFO queue
     * @param k
     * @return
     */
    public double getBeta(int k);
    
    /**
     * Store the values of the FIFO queue
     */
    public void store();
    
    /**
     * Restore the values of the FIFO queue
     */
    public void restore();
}
