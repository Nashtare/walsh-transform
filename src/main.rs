use num::{complex::Complex32, BigInt, Integer, One, Zero};
use primal::StreamingSieve;
use rayon::prelude::*;
use std::f32::consts::PI;

// Computes the extended euclidean algorithm
pub fn egcd<T: Copy + Integer>(a: T, b: T) -> (T, T, T) {
    if a == T::zero() {
        (b, T::zero(), T::one())
    } else {
        let (g, x, y) = egcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

pub fn modinv<T: Copy + Integer>(a: T, m: T) -> Option<T> {
    let (g, x, _) = egcd(a, m);
    if g != T::one() {
        None
    } else {
        Some((x % m + m) % m)
    }
}

pub fn list_to_int(l: &[u64], p: u64) -> usize {
    (l[1] + p * l[0]) as usize
}

pub fn apply_flystel_open(
    z: &[u64; 2],
    beta: &BigInt,
    delta: &BigInt,
    alpha_inv: &BigInt,
    p: u64,
) -> [u64; 2] {
    let p = p.into();
    let x = BigInt::from(z[0]);
    let y = BigInt::from(z[1]);
    let x = x - (beta * z[1] * z[1]);
    let y = y - &x.modpow(&alpha_inv, &p);
    let x = x + beta * &y * &y + delta;

    let x = (x % &p + &p) % &p;
    let y = (y % &p + &p) % p;

    let x = if x.is_zero() {
        0
    } else {
        x.to_u64_digits().1[0]
    };
    let y = if y.is_zero() {
        0
    } else {
        y.to_u64_digits().1[0]
    };

    [x, y]
}

pub fn scalar_product(a: &[u64; 2], b: &[u64; 2], p: u64) -> u64 {
    (a[0] * b[0] + a[1] * b[1]) % p
}

pub fn fourier_transform(
    a: &[u64; 2],
    b: &[u64; 2],
    s_ax: &[Vec<Complex32>],
    s_bfx: &[Vec<Complex32>],
    p: u64,
    boo: bool,
) -> Complex32 {
    let mut result = Complex32::zero();
    for x0 in 0..p {
        for x1 in 0..p {
            let x = [x0, x1];
            let tmp = s_bfx[list_to_int(b, p)][list_to_int(&x, p)];
            if !tmp.is_zero() {
                result += s_ax[list_to_int(a, p)][list_to_int(&x, p)] / tmp;
                if boo {
                    println!("{:?}", result);
                }
            }
        }
    }

    result
}

fn main() {
    (5..10000).into_par_iter().for_each(|prime_index| {
        let start = std::time::Instant::now();
        let mut f = vec![];
        let mut s_ax = vec![];
        let mut s_bfx = vec![];

        let p = StreamingSieve::nth_prime(prime_index) as u64;
        let mut alpha = 2u64;
        let (mut g, _, _) = egcd(alpha as i64, (p - 1) as i64);
        while g != 1 {
            alpha += 1;
            (g, _, _) = egcd(alpha as i64, (p - 1) as i64);
        }
        let mut beta = 2;
        let mut beta_bigint: BigInt = beta.into();
        loop {
            let mut is_generator = true;
            for i in 1..p - 1 {
                if beta_bigint.modpow(&i.into(), &p.into()).is_one() {
                    is_generator = false;
                    break;
                }
            }
            if is_generator {
                break;
            } else {
                beta += 1;
                beta_bigint = beta.into();
            }
        }

        // Unwrapping is guaranteed to work
        let delta = modinv(beta as i64, p as i64).unwrap() as u64;
        let delta_bigint = delta.into();
        // Unwrapping is guaranteed to work
        let alpha_inv = modinv(alpha as i64, (p - 1) as i64).unwrap() as u64 as u32;
        let alpha_inv_bigint = alpha_inv.into();

        let t = 2.0 * PI * Complex32::i() / p as f32;

        // Precomputations
        for x0 in 0..p {
            for x1 in 0..p {
                let x = [x0, x1];
                f.push(apply_flystel_open(
                    &x,
                    &beta_bigint,
                    &delta_bigint,
                    &alpha_inv_bigint,
                    p,
                ));
            }
        }

        for a0 in 0..p {
            for a1 in 0..p {
                let a = [a0, a1];
                let mut sa = vec![];
                let mut sbf = vec![];
                for x0 in 0..p {
                    for x1 in 0..p {
                        let x = [x0, x1];
                        sa.push((t * scalar_product(&a, &x, p) as f32).exp());
                        sbf.push((t * scalar_product(&a, &f[list_to_int(&x, p)], p) as f32).exp());
                    }
                }
                s_ax.push(sa);
                s_bfx.push(sbf);
            }
        }

        // Finding the maximum
        let mut ft_max = 0f32;
        for b0 in 0..p {
            for b1 in 0..p {
                let b = [b0, b1];
                if b == [0, 0] {
                    continue;
                }
                for a0 in 0..p {
                    for a1 in 0..p {
                        let a = [a0, a1];
                        if a == [0, 0] {
                            continue;
                        }
                        let tmp = fourier_transform(&a, &b, &s_ax, &s_bfx, p, false).norm();
                        ft_max = ft_max.max(tmp);
                    }
                }
            }
        }

        let end = std::time::Instant::now();

        println!(
            "p = {:?} | max = {:.2?} (took {:.2?})",
            p,
            ft_max,
            end - start
        );
    })
}
