use rand::Rng;
use std::cell::RefCell;
use std::sync::OnceLock;

struct Config {
    num_lanes: usize,
    wordsize: u64,
}

static CONFIG_CELL: OnceLock<Config> = OnceLock::new();

#[derive(Default)]
struct State {
    cycle: u64,
    rng: rand::rngs::ThreadRng,
    req_bytes: Vec<u64>,
    resp_bytes: Vec<u64>,
    stop: bool,
    finished: bool,
}

thread_local!(static STATE: RefCell<State> = RefCell::new(State::default()));

struct Bundle {
    a_ready: bool,
    a_valid: bool,
    a_address: u64,
    a_size: u32,
    d_ready: bool,
    d_valid: bool,
    d_size: u32,
}

impl Bundle {
    fn a_fire(self: &Self) -> bool {
        self.a_ready && self.a_valid
    }
    fn d_fire(self: &Self) -> bool {
        self.d_ready && self.d_valid
    }
}

#[no_mangle]
pub fn memfuzz_init_rs(num_lanes: i32) {
    CONFIG_CELL.get_or_init(|| Config {
        num_lanes: num_lanes as usize,
        wordsize: 4,
    });
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        s.req_bytes = vec![0; num_lanes as usize];
        s.resp_bytes = vec![0; num_lanes as usize];
    })
}

#[no_mangle]
pub fn memfuzz_generate_rs(
    ptr_a_ready: *const u8,
    ptr_a_valid: *mut u8,
    ptr_a_address: *mut u64,
    _ptr_a_is_store: *mut u8,
    ptr_a_size: *mut u32,
    _ptr_a_data: *mut u64,
    ptr_d_ready: *mut u8,
    ptr_d_valid: *const u8,
    _ptr_d_is_store: *mut u8,
    ptr_d_size: *mut u32,
    inflight: u8,
    ptr_finished: *mut u8,
) {
    let conf = CONFIG_CELL.get().unwrap();

    let vec_a_ready = unsafe { std::slice::from_raw_parts(ptr_a_ready, conf.num_lanes) };
    let vec_a_valid = unsafe { std::slice::from_raw_parts_mut(ptr_a_valid, conf.num_lanes) };
    let vec_a_address = unsafe { std::slice::from_raw_parts_mut(ptr_a_address, conf.num_lanes) };
    let vec_a_size = unsafe { std::slice::from_raw_parts_mut(ptr_a_size, conf.num_lanes) };
    let vec_d_ready = unsafe { std::slice::from_raw_parts_mut(ptr_d_ready, conf.num_lanes) };
    let vec_d_valid = unsafe { std::slice::from_raw_parts(ptr_d_valid, conf.num_lanes) };
    let vec_d_size = unsafe { std::slice::from_raw_parts(ptr_d_size, conf.num_lanes) };
    let finished = unsafe { std::slice::from_raw_parts_mut(ptr_finished, 1) };

    let mut bundles = Vec::with_capacity(conf.num_lanes);
    for i in 0..conf.num_lanes {
        bundles.push(Bundle {
            a_ready: (vec_a_ready[i] != 0),
            a_valid: false, // will be overwritten
            a_address: 0,   // will be overwritten
            a_size: 0,      // will be overwritten
            d_ready: false, // will be overwritten
            d_valid: (vec_d_valid[i] != 0),
            d_size: vec_d_size[i],
        });
    }

    STATE.with(|state| {
        let mut s = state.borrow_mut();
        // s.rng = rand::thread_rng();
        if s.cycle >= 500000 {
            s.stop = true;
        }
        if s.stop && (inflight == 0) {
            s.finished = true;
        }

        generate_a_valid(&mut s, &mut bundles);
        generate_a_addr(&mut s, &mut bundles);
        generate_a_size(conf, &mut s, &mut bundles);
        generate_d_ready(&mut s, &mut bundles);

        // stats
        for i in 0..bundles.len() {
            if bundles[i].a_fire() {
                s.req_bytes[i] += 1 << bundles[i].a_size;
                println!("rust: lane {i} A fire! req_bytes[{i}]={}", s.req_bytes[i]);
            }

            if bundles[i].d_fire() {
                s.resp_bytes[i] += 1 << bundles[i].d_size;
                assert!(bundles[i].d_size == 2, "d_size != 2");
                println!("rust: lane {i} D fire! resp_bytes[{i}]={}", s.resp_bytes[i]);
            }

            assert!(
                s.req_bytes[i] >= s.resp_bytes[i],
                "FAIL: on lane {i}: cumulative request bytes ({}) cannot be less than response bytes ({})!",
                s.req_bytes[i],
                s.resp_bytes[i]
            );
            if s.finished {
                assert!(s.req_bytes[i] == s.resp_bytes[i],
                        "FAIL: on lane {i}: cumulative request bytes ({}) and response bytes ({}) do not match after termination!",
                        s.req_bytes[i], s.resp_bytes[i]);
            }
        }

        finished[0] = if s.finished { 1 } else { 0 };

        s.cycle += 1;
    });

    bundles_to_vecs(
        &bundles,
        vec_a_valid,
        vec_a_address,
        vec_a_size,
        vec_d_ready,
    );
}

// unwrap arrays-of-structs to structs-of-arrays
fn bundles_to_vecs(
    bundles: &[Bundle],
    vec_a_valid: &mut [u8],
    vec_a_address: &mut [u64],
    vec_a_size: &mut [u32],
    vec_d_ready: &mut [u8],
) {
    for i in 0..bundles.len() {
        vec_a_valid[i] = if bundles[i].a_valid { 1 } else { 0 };
        vec_a_address[i] = bundles[i].a_address;
        vec_a_size[i] = bundles[i].a_size;
        vec_d_ready[i] = if bundles[i].d_ready { 1 } else { 0 };
    }
}

fn generate_a_valid(state: &mut State, bundles: &mut [Bundle]) {
    if state.stop {
        for i in 0..bundles.len() {
            bundles[i].a_valid = false;
        }
    } else {
        for i in 0..bundles.len() {
            // 75% chance
            let lottery = state.rng.gen_range(0..4);
            bundles[i].a_valid = lottery >= 1;
        }
    }
}

fn generate_a_addr(state: &mut State, bundles: &mut [Bundle]) {
    let base = 0x80000000u64;
    for i in 0..bundles.len() {
        // bundles[i].a_address = state.rng.gen_range(0..16) * 64 * 7;
        bundles[i].a_address = base + state.rng.gen_range(0..16 * 4)
    }
}

fn generate_a_size(conf: &Config, _state: &mut State, bundles: &mut [Bundle]) {
    for i in 0..bundles.len() {
        bundles[i].a_size = conf.wordsize.ilog2();
    }
}

fn generate_d_ready(state: &mut State, bundles: &mut [Bundle]) {
    for i in 0..bundles.len() {
        if state.stop {
            // wait for all outstanding responses
            bundles[i].d_ready = true;
        } else {
            let lottery = state.rng.gen_range(0..2);
            bundles[i].d_ready = lottery < 1;
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {
        // import_me()
    }
}
