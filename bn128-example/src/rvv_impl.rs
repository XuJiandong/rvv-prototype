use core::mem::transmute;
use rvv::rvv_vector;
use rvv_asm::rvv_asm;
use rvv_simulator_runtime::Uint;

pub type U256 = Uint<4>;
pub type U512 = Uint<8>;

#[macro_export]
macro_rules! U256 {
    ($e: expr) => {
        Uint::<4>($e)
    };
}

#[macro_export]
macro_rules! U512 {
    ($e: expr) => {
        Uint::<8>($e)
    };
}

// implemented by rvv_vector
#[rvv_vector]
pub fn mont_reduce(np1: U256, n: U256, t: U512, bits: usize) -> U256 {
    let t0: U512 = U256::from(t).into(); // low part of `t`, same as `% self.r`, avoid overflow
    let np1_512: U512 = U512::from(np1);
    let m_512: U512 = U256::from(t0 * np1_512).into(); // `% self.r`
    let n_512: U512 = U512::from(n);
    let bits_512: U512 = U512::from(bits);
    let u: U512 = (t + m_512 * n_512) >> bits_512; // `/ self.r`
    if u >= n_512 {
        U256::from(u - n_512)
    } else {
        U256::from(u)
    }
}

#[rvv_vector]
pub fn mont_multi(np1: U256, n: U256, x: U256, y: U256, bits: usize) -> U256 {
    let x_512: U512 = x.into();
    let y_512: U512 = y.into();

    let xy: U512 = x_512 * y_512;
    mont_reduce(np1, n, xy, bits)
}

#[inline(never)]
pub fn mont_multi_asm(np1: &[U256], n: &[U256], x: &[U256], y: &[U256], res: &mut [U256]) {
    let len = x.len() as u64;
    // not loop version
    debug_assert!(len <= 32);
    // registers allocation:
    // x -> v4
    // y -> v8
    // np1 -> v12
    // n -> v16
    // m -> v20
    // t -> v24 = x*y
    // u -> v24 (512 bits)
    // high 256 bits part of x*y -> v28
    unsafe {
        rvv_asm!(
            "mv t1, {len}",
            "vsetvli t2, t1, e256, m4",

            "mv t0, {x}",
            "vle256.v v4, (t0)",

            "mv t0, {y}",
            "vle256.v v8, (t0)",

            "mv t0, {np1}",
            "vle256.v v12, (t0)",

            "mv t0, {n}",
            "vle256.v v16, (t0)",

            "vmul.vv v24, v4, v8",
            "vmul.vv v20, v24, v12",

            // vd[i] = +(vs1[i] * vs2[i]) + vd[i]
            // vd: SEW*2, vs1: SEW, vs2: SEW
            "vwmaccu.vv v24, v20, v16",

            // shift right and narrowing
            "addi t0, x0, 256",
            "vnsra.wx v24, v24, t0",

            // high 256 bits of x*y
            "vmulhu.vv v28, v4, v8",
            "vadd.vv v24, v24, v28",

            // u >= n
            "vmsleu.vv v0, v16, v24",
            // u = u - n
            "vsub.vv v24, v24, v16, v0.t",

            "mv t0, {res}",
            "vse256.v v24, (t0)",

            np1 = in (reg) np1.as_ptr(),
            n =  in (reg) n.as_ptr(),
            x = in (reg) x.as_ptr(),
            y = in (reg) y.as_ptr(),
            res = in (reg) res.as_ptr(),
            len = in (reg) len,
        );
    }
}

#[inline(always)]
fn from_u128pair(n: &[u128; 2]) -> U256 {
    let buf: &[u64; 4] = unsafe { transmute(n) };
    Uint::<4>(buf.clone())
}

pub fn mul_reduce_internal(
    this: &mut [u128; 2],
    by: &[u128; 2],
    modulus: &[u128; 2],
    inv: u128,
    inv_high: u128,
) {
    let x: U256 = from_u128pair(this);
    let y: U256 = from_u128pair(by);
    let n: U256 = from_u128pair(modulus);
    let np1: U256 = from_u128pair(&[inv, inv_high]);

    if cfg!(feature = "use_rvv_vector") {
        let Uint::<4>(ref result) = mont_multi(np1, n, x, y, 256);
        this[0] = result[0] as u128 | (result[1] as u128) << 64;
        this[1] = result[2] as u128 | (result[3] as u128) << 64;
    } else {
        let mut result = [U256::from(0)];
        mont_multi_asm(&[np1], &[n], &[x], &[y], &mut result);
        let Uint::<4>(ref result) = result[0];
        this[0] = result[0] as u128 | (result[1] as u128) << 64;
        this[1] = result[2] as u128 | (result[3] as u128) << 64;
    }
}

pub fn bench_mont() {
    let mut this = [0x1234567890ABCDEF1234567890ABCDEF, 0x111111111111111111];
    let by = [0x1234567891111111, 0x12345678922222222];
    let modulus = [0x123456789001, 0x1234567892];
    let inv = 0x123456789;
    let inv_high = 0x12345678;
    for _ in 0..72650 {
        mul_reduce_internal(&mut this, &by, &modulus, inv, inv_high);
    }
}
