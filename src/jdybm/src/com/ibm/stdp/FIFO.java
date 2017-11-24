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
