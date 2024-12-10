use crate::structure::{Neuron, Synapse};

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
}