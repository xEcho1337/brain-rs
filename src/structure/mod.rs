mod neuron;
mod synapse;

pub struct Neuron {
    pub synapses: Vec<Synapse>,
    pub delta: f64,
    pub value: f64,
    pub bias: f64,
}

pub struct Synapse {
    pub input_neuron: Neuron,
    pub output_neuron: Neuron,
    pub id: usize,
    pub weight: f64
}