use std::sync::atomic::{AtomicUsize, Ordering};

use rand::random;

use crate::structure::{Neuron, Synapse};

pub static SYNAPSE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Synapse {
    pub fn new(input_neuron: Neuron, output_neuron: Neuron, bound: f64) -> Self {
        let id = SYNAPSE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            input_neuron,
            output_neuron,
            id,
            weight: (random() * 2 * bound) - bound
        }
    }

    pub fn get_total_synapses() -> usize {
        SYNAPSE_ID_COUNTER.load(Ordering::SeqCst)
    }
}