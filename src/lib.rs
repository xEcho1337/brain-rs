use crate::activation::Activations;
use crate::layer::DenseLayer;
use crate::model::Model;

mod model;
mod utils;
mod layer;
mod optimizer;
mod activation;
mod structure;

fn test() {
    let model = Model::new(vec![
        DenseLayer::new(2, Activations::Linear),
        DenseLayer::new(16, Activations::ReLU),
        DenseLayer::new(16, Activations::ReLU),
        DenseLayer::new(1, Activations::Sigmoid),
    ]);
}