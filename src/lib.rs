use jni::objects::{JDoubleArray, JObject, ReleaseMode};
use jni::sys::{jlong, jdouble, jdoubleArray};
use jni::JNIEnv;
use std::sync::Arc;

#[repr(C)]
pub struct NativeVector {
    pub data: Arc<Vec<f64>>,
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_convolute(
    env: JNIEnv,
    _class: jni::objects::JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> jdoubleArray {
    let a_len = env.get_array_length(&a).unwrap();
    let b_len = env.get_array_length(&b).unwrap();

    let mut b_vec = vec![0.0; b_len as usize];
    let mut a_vec = vec![0.0; a_len as usize];

    env.get_double_array_region(a, 0, &mut a_vec).unwrap();
    env.get_double_array_region(b, 0, &mut b_vec).unwrap();

    let a_length = a_len as usize;
    let b_length = b_len as usize;

    let mut result = vec![0.0; a_length + b_length - 1];

    for i in 0..a_length {
        for j in 0..b_length {
            result[i + j] += a_vec[i] * b_vec[j];
        }
    }

    let result_array = env.new_double_array(result.len() as i32).unwrap();
    env.set_double_array_region(&result_array, 0, &result).unwrap();

    **result_array
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_sum2(
    env: JNIEnv,
    _class: jni::objects::JClass,
    input: JDoubleArray,
) -> jdouble {
    let array_length = env.get_array_length(&input).expect("Errore nel recuperare la lunghezza dell'array");
    let mut buffer = vec![0.0; array_length as usize];
    env.get_double_array_region(input, 0, &mut buffer)
        .expect("Errore nel copiare i dati dell'array");

    let mut total_sum = 0.0;

    for _ in 0..100_000 {
        total_sum += buffer.iter().sum::<f64>();
    }

    total_sum
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_init(
    _env: JNIEnv,
    _class: JObject,
    size: i32,
) -> jlong {
    let vector = NativeVector {
        data: Arc::new(vec![0.0; size as usize]),
    };
    Box::into_raw(Box::new(vector)) as jlong
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_initWithData(
    env: JNIEnv,
    _class: JObject,
    data: JDoubleArray,
    _length: i32,
) -> jlong {
    let length = env.get_array_length(&data).unwrap() as usize;

    let mut result = vec![0.0; length];

    env.get_double_array_region(data, 0, &mut result).unwrap();

    let vector = NativeVector {
        data: Arc::new(result),
    };

    Box::into_raw(Box::new(vector)) as jlong
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_length(
    env: JNIEnv,
    _class: JObject,
    vector_ptr: jlong,
) -> jdouble {
    let vector = unsafe { &*(vector_ptr as *const NativeVector) };
    vector.data.len() as f64
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_sum(
    env: JNIEnv,
    _class: JObject,
    vector_ptr: jlong,
) -> jdouble {
    let vector = unsafe { &*(vector_ptr as *const NativeVector) };

    let len = vector.clone().data.len();
    let sum= 0;
    println!("{}", sum);
    println!("Len: {}", len);

    return sum as jdouble
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_normalize(
    env: JNIEnv,
    _class: JObject,
    vector_ptr: jlong,
) {
    let vector = unsafe { &mut *(vector_ptr as *mut NativeVector) };

    let mut data = Arc::make_mut(&mut vector.data);

    let magnitude = data.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if magnitude > 0.0 {
        for elem in data.iter_mut() {
            *elem /= magnitude;
        }
    }
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_scale(
    env: JNIEnv,
    _class: JObject,
    vector_ptr: jlong,
    value: jdouble,
) {
    let vector = unsafe { &mut *(vector_ptr as *mut NativeVector) };

    let mut data = Arc::make_mut(&mut vector.data);

    for elem in data.iter_mut() {
        *elem *= value;
    }
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_distance(
    env: JNIEnv,
    _class: JObject,
    vec1_ptr: jlong,
    vec2_ptr: jlong,
) -> jdouble {
    let vec1 = unsafe { &*(vec1_ptr as *const NativeVector) };
    let vec2 = unsafe { &*(vec2_ptr as *const NativeVector) };

    if vec1.data.len() != vec2.data.len() {
        panic!("Vectors must have the same length");
    }

    let mut sum = 0.0;

    for i in 0..vec1.data.len() {
        let diff = vec1.data[i] - vec2.data[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

#[no_mangle]
pub extern "C" fn Java_net_echo_brain4j_utils_NativeVector_free(
    env: JNIEnv,
    _class: JObject,
    vector_ptr: jlong,
) {
    unsafe {
        let _ = Box::from_raw(vector_ptr as *mut NativeVector);
    }
}