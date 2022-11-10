use std::error::Error; //manejar errores
use std::fs::File; // leer archivos
use std::process; // terminar procesos
use std::time::{Instant};
use rand::Rng;

fn read_csv() -> Result<Vec<Vec<u8>>, Box<dyn Error>>{ // -> tipo de dato a devolver, Result devuelve vacio si no hay error, devuelve la funcion si hay un error
    let data = File::open("train.csv")?;
    let mut rdr = csv::Reader::from_reader(data);
    let mut data_matrix: Vec<Vec<u8>> = Vec::new();
    for result in rdr.deserialize(){
     let record: Vec<u8> = result?;
        data_matrix.push(record);
    }
    Ok(data_matrix)
}
fn matrix_transpose(matrix: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut transpose_matrix = vec![Vec::with_capacity(matrix.len()); matrix[0].len()];
    for row in matrix {
        for column in 0..row.len(){
            transpose_matrix[column].push(row[column]);
        }
    }
    transpose_matrix
}
fn matrix_to_string(matrix: &Vec<Vec<f64>>) -> String {
    matrix.iter().fold("".to_string(), |a, r| {
        a + &r
            .iter()
            .fold("".to_string(), |b, e| b + "\t" + &e.to_string())
            + "\n"
    })
}

fn init_variables() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>){
    let mut hidden_layer1: Vec<Vec<f64>> = Vec::new();
    let mut hidden_layer2: Vec<Vec<f64>> = Vec::new();
    let mut bias_layer1: Vec<Vec<f64>> = Vec::new();
    let mut bias_layer2: Vec<Vec<f64>> = Vec::new();
    let mut rng = rand::thread_rng();
    for _row in 0..10{
        let mut vector: Vec<f64> = Vec::new();
        for _column in 0..784{
           let mut random : f64 = rng.gen();
           random -= 0.5;
            vector.push(random);
        }
        hidden_layer1.push(vector);
    }
    for _row in 0..10{
        let mut vector: Vec<f64> = Vec::new();
        for _column in 0..1{
           let mut random : f64 = rng.gen();
           random -= 0.5;
            vector.push(random);
        }
       bias_layer1.push(vector);
    }
    for _row in 0..10{
        let mut vector: Vec<f64> = Vec::new();
        for _column in 0..10{
           let mut random : f64 = rng.gen();
           random -= 0.5;
            vector.push(random);
        }
       hidden_layer2.push(vector);
    }
    for _row in 0..10{
        let mut vector: Vec<f64> = Vec::new();
        for _column in 0..1{
           let mut random : f64 = rng.gen();
           random -= 0.5;
            vector.push(random);
        }
       bias_layer2.push(vector);
    }
    (hidden_layer1,bias_layer1, hidden_layer2, bias_layer2)
}

fn ReLu(){

}
fn forward_prop(hl1 : Vec<Vec<f64>>,b1: Vec<Vec<f64>>, hl2: Vec<Vec<f64>>, b2: Vec<Vec<f64>>, X: Vec<Vec<f64>>){
   let z1 : Vec<Vec<f64>> = Vec::new();
   for h1_row in 0..hl1.len(){ // n son filas de hl1
    for X_column in 0..X[0].len(){
        for h1_column in 0..hl1[0].len(){

        } // m son columnas de hl1
    
    }
   }

}
fn gradient_descent(pixe : Vec<Vec<u8>>, y: Vec<Vec<u8>>, alpha: f64, iterations: u16){
    let var_array: (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) = init_variables();
    let hidden_layer1 = var_array.0;
    let bias_layer1 = var_array.1;
    let hidden_layer2 = var_array.2;
    let bias_layer2 = var_array.3;
    for i in 0..iterations{

    }


}
fn dot_product(matrix1 : Vec<Vec<f64>>, matrix2 : Vec<Vec<u8>>){
    let mut matrix3 : Vec<Vec<f64>> = Vec::new();
    for matrix1_row in 0..matrix1.len(){ 
        let mut vec :Vec<f64> = Vec::new();
        for matrix2_column in 0..matrix2[0].len(){
            let mut a:f64 = 0.0;
            for matrix1_column in 0..matrix1[0].len(){
                a += matrix1[matrix1_row][matrix1_column] * matrix2[matrix1_column][matrix2_column] as f64;
            }
            vec.push(a) // m son columnas de hl1
        }
        matrix3.push(vec);
       }
}


fn main(){
    let start = Instant::now();
    let mut data: Vec<Vec<u8>> = Vec::new();
    if let Err(err) = read_csv(){
        println!("error:{}",err);
        process::exit(1);
    } 
    else{
        data = read_csv().unwrap_or_default();
    } 
    let m: u16; //filas
    let mut n: u16 = 0; // columnas
    m = data.len() as u16;
    n = data[0].len() as u16;
    let mut test_data: Vec<Vec<u8>> = Vec::new();
    for i in 0..1000{
       test_data.push(data[i].clone())
    }
    test_data = matrix_transpose(test_data);
    let digits_test_data = &test_data[0];// Columna de los resultados de cada ejemplo
    let mut pixels_test_data : Vec<Vec<u8>> = Vec::new();
    for i in 1..n as usize{
        pixels_test_data.push(test_data[i].clone())
    }
    let mut train_data: Vec<Vec<u8>> = Vec::new();
    for i in 1000..m as usize{
        train_data.push(data[i].clone())
    }
    train_data = matrix_transpose(train_data);
    let digits_train_data = &train_data[0]; // Columna de los resultados de cada ejemplo
    let mut pixels_train_data : Vec<Vec<u8>> = Vec::new(); // pixeles
    for i in 1..n as usize{
        pixels_train_data.push(train_data[i].clone())
    }

    let var_array: (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) = init_variables();
    let xd = var_array.0;
    let duration = start.elapsed();
    println!("El tiempo de entrenamiento fue de {:?}", duration);
}
