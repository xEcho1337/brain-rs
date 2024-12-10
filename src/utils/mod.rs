use std::fmt::Display;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    pub fn from_data(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn of(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    pub fn random(size: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            data: (0..size).map(|_| rng.gen()).collect(),
        }
    }

    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    pub fn length_squared(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum()
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn normalize_squared(&mut self) -> &mut Self {
        let length_squared = self.length_squared();
        self.data.iter_mut().for_each(|x| *x /= length_squared);
        self
    }

    pub fn normalize(&mut self) -> &mut Self {
        let length = self.length();
        self.data.iter_mut().for_each(|x| *x /= length);
        self
    }

    pub fn distance_squared(&self, other: &Vector) -> f64 {
        if self.data.len() != other.data.len() {
            panic!("Vectors must be of the same length.");
        }
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum()
    }

    pub fn distance(&self, other: &Vector) -> f64 {
        self.distance_squared(other).sqrt()
    }

    pub fn convoluted(&self, other: &Vector) -> Vector {
        let mut result = vec![0.0; self.data.len() + other.data.len() - 1];
        for (i, &a) in self.data.iter().enumerate() {
            for (j, &b) in other.data.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        Vector::from_data(result)
    }

    pub fn add(&mut self, other: &Vector) -> &mut Self {
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(a, &b)| *a += b);
        self
    }

    pub fn scale(&mut self, value: f64) -> &mut Self {
        self.data.iter_mut().for_each(|x| *x *= value);
        self
    }

    pub fn fill(&mut self, value: f64) -> &mut Self {
        self.data.iter_mut().for_each(|x| *x = value);
        self
    }

    pub fn fill_with<F>(&mut self, mut function: F) -> &mut Self
    where
        F: FnMut() -> f64,
    {
        self.data.iter_mut().for_each(|x| *x = function());
        self
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    pub fn variance_with_mean(&self, mean: f64) -> f64 {
        self.data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.data.len() as f64
    }

    pub fn variance(&self) -> f64 {
        self.variance_with_mean(self.mean())
    }

    pub fn to_array(&self) -> &[f64] {
        &self.data
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{:?}", self.data))
    }
}
