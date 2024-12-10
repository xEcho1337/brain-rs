use utils::Vector;

use crate::activation::linear::LinearActivation;
use crate::activation::relu::ReLUActivation;
use crate::activation::sigmoid::SigmoidActivation;
use crate::initialization::WeightInitialization;
use crate::layer::DenseLayer;
use crate::loss::LossFunctions;
use crate::model::Model;
use crate::optimizer::{Adam, Optimizer};

pub mod activation;
pub mod initialization;
pub mod layer;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod structure;
pub mod utils;
pub mod training;

fn test() {
    //let inputLayer = DenseLayer::new(2, LinearActivation::default());
    //let hiddenLayer1 = DenseLayer::new(16, ReLUActivation::default());
    //let hiddenLayer2 = DenseLayer::new(16, ReLUActivation::default());
    //let outputLayer = DenseLayer::new(1, SigmoidActivation::default());

    let mut model = Model::new(vec![
        DenseLayer::new(2, LinearActivation::default()),
        DenseLayer::new(16, ReLUActivation::default()),
        DenseLayer::new(16, ReLUActivation::default()),
        DenseLayer::new(1, SigmoidActivation::default())
    ]);

    model.compile(
        WeightInitialization::He,
        LossFunctions::CrossEntropy,
        Adam::new(0.001),
    );

    let input = Vector::of(&[0., 1.]);
    let output = Vector::of(&[0., 1.]);

    // let row = DataRow::new(input.clone(), output.clone());

    let predicted: Vector = model.predict(input.clone());

    println!("{}", predicted)
}
