extern crate ndarray;
use std::error::Error; //manejar errores
use std::fs::File; // leer archivos
use std::process; // terminar procesos
use std::time::{Instant};
use rand::Rng;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
fn read_csv(path_to_file: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    Ok(reader.deserialize_array2((42000, 785))?)
}


fn init_variables() -> (Array2<f64>, Array2<f64>,Array2<f64>, Array2<f64>){
       let W1 = Array::random((10,784), Uniform::new(0.0, 1.0))-0.5;
       let b1 = Array::random((10,1), Uniform::new(0.0, 1.0))-0.5;
       let W2 = Array::random((10,10), Uniform::new(0.0, 1.0))-0.5;
       let b2 = Array::random((10,1), Uniform::new(0.0, 1.0))-0.5;
    (W1,b1,W2,b2)
}
fn ReLu(X: &Array2<f64>) -> Array2<f64>{
    X.mapv(|x| x.max(0.0))
}
fn softmax(X:&Array2<f64>) -> Array2<f64>{
    let X_exp = X.mapv(|x| x.exp());
    let X_sum = X_exp.sum_axis(Axis(0));
    let X_softmax = X_exp/X_sum;
    X_softmax
}


fn forward_prop(W1 : Array2<f64>,b1: Array2<f64>, W2: Array2<f64>, b2: Array2<f64>, X: Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64> ,Array2<f64> ){
    let Z1 = W1.dot(&X) + &b1;
    let A1 = ReLu(&Z1);
    let Z2 = W2.dot(&A1) + &b2;
    let  A2 = softmax(&Z2);
    (Z1,A1,Z2,A2)
}
fn Relu_derivative(Z: &Array2<f64>) -> Array2<f64>{
    Z.mapv(|x| if x>0.0 {1.0} else {0.0})
}
fn one_hot(Y: &Array1<f64>) -> Array2<f64>{
    let mut Y_one_hot = Array::zeros((Y.len(), 10));
    for i in 0..Y.len(){
        Y_one_hot[[i, Y[i] as usize]] = 1.0;
    }
    Y_one_hot.t().to_owned()
}

fn backward_prop(Z1: Array2<f64>, A1: Array2<f64>, Z2: Array2<f64>, A2: Array2<f64>, W1: Array2<f64>, W2: Array2<f64>, X: Array2<f64>, Y: Array1<f64>)->(Array2<f64>,Array1<f64>,Array2<f64>,Array1<f64>){
    let one_hot_Y = one_hot(&Y);
    let dZ2 = A2 - one_hot_Y;
    let dW2 = dZ2.dot(&A1.t())/42000.0;
    let db2 = dZ2.sum_axis(Axis(1))/42000.0;
    let dZ1 = W2.t().dot(&dZ2)*Relu_derivative(&Z1);
    let dW1 = dZ1.dot(&X.t())/42000.0;
    let db1 = dZ1.sum_axis(Axis(1))/42000.0;
    (dW1,db1,dW2,db2)
 

}
fn update_params(W1: Array2<f64>, b1: Array2<f64>, W2: Array2<f64>, b2: Array2<f64>, dW1: Array2<f64>, db1: Array1<f64>, dW2: Array2<f64>, db2: Array1<f64>, learning_rate: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>){
    let W1 = W1 - learning_rate*dW1;
    let b1 = b1 - learning_rate*db1;
    let W2 = W2 - learning_rate*dW2;
    let b2 = b2 - learning_rate*db2;
    (W1,b1,W2,b2)
}
fn get_predictions(A2: Array2<f64>) -> Array1<f64>{
    let mut predictions = Array::zeros(A2.shape()[1]);
    for i in 0..A2.shape()[1]{
        let mut max = 0.0;
        let mut max_index = 0;
        for j in 0..A2.shape()[0]{
            if A2[[j,i]] > max{
                max = A2[[j,i]];
                max_index = j;
            }
        }
        predictions[i] = max_index as f64;
    }
    predictions
}
fn get_accuracy(predictions: Array1<f64>, Y: Array1<f64>) -> f64{
    let mut correct = 0.0;
    for i in 0..predictions.len(){
        if predictions[i] == Y[i]{
            correct += 1.0;
        }
    }
    correct/predictions.len() as f64
}
fn gradient_descent(X: Array2<f64>, Y: Array1<f64>, learning_rate: f64, iterations: i32) ->(Array2<f64>,Array2<f64>,Array2<f64>,Array2<f64>){
    let (W1,b1,W2,b2) = init_variables();
    for i in 0..iterations{
        let (Z1,A1,Z2,A2) = forward_prop(W1,b1,W2,b2,X);
        let (dW1,db1,dW2,db2) = backward_prop(Z1,A1,Z2,A2,W1,W2,X,Y);
        let (W1,b1,W2,b2) = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate);
        if i%10 == 0{
            println!("Iteration: {}", i);
            println!("Accuracy: {}", get_accuracy(get_predictions(A2),Y));
        }
    }
    (W1,b1,W2,b2)
}
fn main(){
    let start = Instant::now();
    let data= read_csv("train.csv").unwrap();
    let m = data.shape()[0];
    let n = data.shape()[1];

    let binding = data.slice(s![0..1000, ..]);
    let data_dev = binding.t();
    let Y_dev = data_dev.slice(s![0, ..]).to_owned();
    let mut X_dev = data_dev.slice(s![1..n, ..]).to_owned();
    X_dev = X_dev.mapv(|x| x/255.0);
    let binding = data.slice(s![1000..m, ..]);
    let data_train = binding.t();
    let Y_train = data_train.slice(s![0, ..]).to_owned();
    let mut X_train = data_train.slice(s![1..n, ..]).to_owned();
    X_train  = X_train.mapv(|x| x/255.0);

    let (W1,b1,W2,b2) = gradient_descent(X_train,Y_train,0.1,1000);


    let duration = start.elapsed();
    println!("El tiempo de entrenamiento fue de {:?}", duration);

}
