/// This module implements an exhaustive search of the maximum Fourier
/// transform applied to the Anemoi S-box (Flystel) over bidimensional
/// vectors of a prime field Fp, in order to support a conjecture
/// expressed in the Anemoi/Jive paper:
/// https://eprint.iacr.org/2022/840.pdf
///
/// It is based off a sequential SAGEMATH implementation produced by
/// Clemence Bouvier (https://who.rocq.inria.fr/Clemence.Bouvier/).
use num::{complex::Complex32, BigInt, Integer, One, Zero};
use primal::StreamingSieve;
use std::cmp::min;
use std::f32::consts::PI;

// Needed for parallelisation
use rayon::prelude::*;
use std::sync::RwLock;

//////////////////////////////////////////
///////////// HELPER METHODS /////////////
//////////////////////////////////////////

/// Computes the extended Euclidean algorithm
pub fn egcd<T: Copy + Integer>(a: T, b: T) -> (T, T, T) {
    if a == T::zero() {
        (b, T::zero(), T::one())
    } else {
        let (g, x, y) = egcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

/// Computes the modular inverse of a mod m
pub fn modinv<T: Copy + Integer>(a: T, m: T) -> Option<T> {
    let (g, x, _) = egcd(a, m);
    if g != T::one() {
        None
    } else {
        Some((x % m + m) % m)
    }
}

/// Converts an integer to a list giving its decomposition
/// in base p.
///
/// The method assumes that x fits in [0, p^2) as it outputs
/// a bi-dimensional vector.
pub fn int_to_list(x: u64, p: u64) -> [u64; 2] {
    [x / p, x % p]
}

/// Converts an bi-dimensional array to an integer.
///
/// The method assumes that the coordinates of the array fit
/// in the range [0, p).
pub fn list_to_int(l: &[u64; 2], p: u64) -> usize {
    (l[1] + p * l[0]) as usize
}

//////////////////////////////////////////
////////////// CORE METHODS //////////////
//////////////////////////////////////////

/// Applies a Flystel S-Box defined in the Anemoi/Jive
/// paper (https://eprint.iacr.org/2022/840.pdf), i.e.
/// given a vector z of two elements x and y, perform:
/// - x <- x - (beta*y^2)
/// - y <- y - x^a_inv
/// - x <- x + beta*y^2 + delta
///
/// and returns the new vector z' = [x,y].
///
/// Beta, delta and a_inv are the Flystel parameters, and all arithmetic
/// is performed modulo some prime p.
///
/// The method assumes that the constants have been correctly computed,
/// i.e. that:
/// - beta is the smallest multiplicative generator of the prime fielf
/// of characteristic p;
/// - delta is the inverse of beta (e.g. delta = beta^(-1));
/// - alpha_inv is the inverse of alpha, the smallest element such that
/// gcd(alpha, p-1) = 1.
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

    // Perform the Flystel S-Box on x and y
    let x = x - (beta * z[1] * z[1]);
    let y = y - &x.modpow(&alpha_inv, &p);
    let x = x + beta * &y * &y + delta;

    // Reduce the results in the range [0, p)
    let x = (x % &p + &p) % &p;
    let y = (y % &p + &p) % p;

    // The internal representation of a zero BigInt is
    // empty hence handling the edge case manually.
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

/// Computes the scalar product of two 2-dimensional vectors modulo a prime p.
pub fn scalar_product(a: &[u64; 2], b: &[u64; 2], p: u64) -> u64 {
    (a[0] * b[0] + a[1] * b[1]) % p
}

/// Computes the Fourier transform between two vectors of dimension 2 a and b
/// in a prime field of characteristic p.
pub fn fourier_transform(
    a: usize,
    b: usize,
    s_ax: &[Vec<Complex32>],
    s_bfx: &[Vec<Complex32>],
) -> Complex32 {
    let mut result = Complex32::zero();
    for (index, tmp) in s_bfx[b].iter().enumerate() {
        result += s_ax[a][index] / tmp;
    }

    result
}

fn main() {
    // We start at p = 11
    let mut prime_index = 5;

    loop {
        let start = std::time::Instant::now();

        let p = StreamingSieve::nth_prime(prime_index) as u64;

        // Generate the Flystel parameters
        let mut alpha = 2u64;
        let (mut g, _, _) = egcd(alpha as i64, (p - 1) as i64);
        while g != 1 {
            alpha += 1;
            (g, _, _) = egcd(alpha as i64, (p - 1) as i64);
        }
        // Unwrapping is guaranteed to work
        let alpha_inv = modinv(alpha as i64, (p - 1) as i64).unwrap() as u64 as u32;
        let alpha_inv_bigint = alpha_inv.into();

        let mut beta = 2;
        let mut beta_bigint: BigInt = beta.into();
        loop {
            // There are faster checks to ensure that beta is a generator of the multiplicative
            // subgroup, but the running time of this naive approach is negligible compared to
            // the main routine anyway...
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

        // Additional variables
        let t = 2.0 * PI * Complex32::i() / p as f32;
        let num_threads = rayon::current_num_threads() as u64;
        let chunk_size = (p * p) / num_threads;

        // Pre-allocate the vectors of values we will need to generate
        let mut f = Vec::with_capacity((p * p) as usize);
        let mut s_ax = Vec::with_capacity((p * p) as usize);
        let mut s_bfx = Vec::with_capacity((p * p) as usize);

        // Phase 1: Precomputation of all the possible S-Box outputs
        // We iterate over all 2-dimensional vectors over Fp (p^2 cases)
        //
        // Parallelisation note: the running time of this phase is negligible
        // compared to the other phases, hence it is kept sequential.
        for x0 in 0..p {
            for x1 in 0..p {
                f.push(apply_flystel_open(
                    &[x0, x1],
                    &beta_bigint,
                    &delta_bigint,
                    &alpha_inv_bigint,
                    p,
                ));
            }
        }

        // Phase 2: Precomputation of all the possible
        // We iterate over all 2-dimensional vectors a over Fp (p^2 cases),
        // and for each, compute
        // - exp((2iπ * <a|x>)/p)
        // - exp((2iπ * <a|f[x]>)/p)
        //
        // where x covers all 2-dimensional vectors over Fp, <a|x> represents
        // the scalar product (in Fp) of the vectors a and x, and f[x]
        // corresponds to the output of the flystel S-Box applied on x, which
        // has been precomputed during Phase 1.
        //
        // Parallelisation note: the running time of this phase is negligible
        // compared to the search phase, however it starts taking a consequent
        // time quickly when p grows (it is running in O(p^4)), hence it would
        // be beneficial to parallelize it to some extent.
        for a0 in 0..p {
            for a1 in 0..p {
                let a = [a0, a1];
                let mut sa = Vec::with_capacity((p * p) as usize);
                let mut sbf = Vec::with_capacity((p * p) as usize);
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

        // Phase 3: Exhaustive maxima search
        // We iterate over all pairs of 2-dimensional vectors a and b over Fp
        // (p^4 cases), and for each, compute the Fourier transform W of <b|f>
        // applied on a.
        // The output of this phase is the maximum Fourier transform value
        // obtained among all couples of vectors a and b.
        //
        // Parallelisation note: the running time of this phase is much more
        // consequent than the two previous phases, because we cover all pairs
        // of 2-dimensional vectors (p^4 possibilities), and for each, iterate
        // over p^2 elements to divide the exponents and add the result to the
        // running accumulator, ending in a total complexity of O(p^6).
        // To reduce the running cost, we split the first loop (iterating over
        // the first 2-dimensional vector into N parallel instances, with N the
        // number of available processes.
        // The shared global Fourier transform maxima is behind a RwLock, such
        // that each thread will only block the other ones once, at the end of
        // its loop.
        let global_ft_max = RwLock::new(0f32);
        (0..num_threads).into_par_iter().for_each(|i| {
            let mut local_ft_max = 0f32;

            for b in i * chunk_size..min((i + 1) * chunk_size + 1, p * p) {
                if b == 0 {
                    continue;
                }
                for a in 0..p * p {
                    let tmp = fourier_transform(a as usize, b as usize, &s_ax, &s_bfx).norm();
                    local_ft_max = local_ft_max.max(tmp);
                }
            }

            // We first read the global maxima, because we do not want the thread
            // to be blocking others in case the local maxima is smaller than the
            // global one.
            let current_global_ft_max = global_ft_max.read().unwrap().clone();
            // If the local maxima is greater, then we update the global maxima.
            if current_global_ft_max < local_ft_max {
                *global_ft_max.write().unwrap() = local_ft_max;
            }
        });

        let end = std::time::Instant::now();

        println!(
            "p = {:?} | alpha = {:?} | max = {:.2?}\t (took {:.2?})",
            p,
            alpha,
            global_ft_max.read().unwrap(),
            end - start
        );

        prime_index += 1;
    }
}
