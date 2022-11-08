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
    for _column in &data[0]{
        n+=1;
    }
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
    let mut pixels_train_data : Vec<Vec<u8>> = Vec::new();
    for i in 1..n as usize{
        pixels_train_data.push(train_data[i].clone())
    }
    println!("{}",pixels_train_data.len());
   let mut B: (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) = init_variables();

    println!("{:?}",&B.2);
    let duration = start.elapsed();

    println!("El tiempo de entrenamiento fue de {:?}", duration);
}
