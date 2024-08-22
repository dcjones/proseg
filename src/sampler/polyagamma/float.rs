use libm::{erfc, erfcf, lgamma_r, lgammaf_r};
use num_traits::{cast::NumCast, float};
use std::ops::{AddAssign, MulAssign, SubAssign};

// Wrapping the num_traits Float trait to add more special functions.
pub trait Float: float::Float + AddAssign + SubAssign + MulAssign {
    // fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn lgamma(self) -> Self;
    fn as_usize(self) -> usize {
        <usize as NumCast>::from(self).unwrap()
    }

    fn as_f64(self) -> f64 {
        <f64 as NumCast>::from(self).unwrap()
    }

    const TAU: Self;
    const SQRT_2: Self;
}

impl Float for f32 {
    // fn erf(self) -> Self {
    //     erff(self)
    // }

    fn erfc(self) -> Self {
        erfcf(self)
    }

    fn lgamma(self) -> Self {
        lgammaf_r(self).0
    }

    const TAU: Self = std::f32::consts::TAU;
    const SQRT_2: Self = std::f32::consts::SQRT_2;
}

impl Float for f64 {
    // fn erf(self) -> Self {
    //     erf(self)
    // }

    fn erfc(self) -> Self {
        erfc(self)
    }

    fn lgamma(self) -> Self {
        lgamma_r(self).0
    }

    const TAU: Self = std::f64::consts::TAU;
    const SQRT_2: Self = std::f64::consts::SQRT_2;
}
