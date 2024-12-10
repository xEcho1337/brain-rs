use crate::activation::Activations;
use crate::initialization::WeightInitialization;
use crate::layer::DenseLayer;
use crate::model::Model;

mod model;
mod utils;
mod layer;
mod optimizer;
mod activation;
mod structure;
mod initialization;
mod loss;

fn test() {
    let mut _model = Model::new(vec![
        DenseLayer::new(2, Activations::Linear),
        DenseLayer::new(16, Activations::ReLU),
        DenseLayer::new(16, Activations::ReLU),
        DenseLayer::new(1, Activations::Sigmoid),
    ]);

    _model.compile(WeightInitialization::He)
}