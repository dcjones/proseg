
use num_traits::{cast::NumCast, float};
use std::ops::{AddAssign, SubAssign, MulAssign};
use libm::{erf, erff, erfc, erfcf, lgamma_r, lgammaf_r};


// Wrapping the num_traits Float trait to add more special functions.
pub trait Float: float::Float + AddAssign + SubAssign + MulAssign {
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn lgamma(self) -> Self;
    fn as_usize(self) -> usize {
        return <usize as NumCast>::from(self).unwrap();
    }

    fn as_f64(self) -> f64 {
        return <f64 as NumCast>::from(self).unwrap();
    }
}

impl Float for f32 {
    fn erf(self) -> Self {
        return erff(self);
    }

    fn erfc(self) -> Self {
        return erfcf(self);
    }

    fn lgamma(self) -> Self {
        return lgammaf_r(self).0;
    }
}

impl Float for f64 {
    fn erf(self) -> Self {
        return erf(self);
    }

    fn erfc(self) -> Self {
        return erfc(self);
    }

    fn lgamma(self) -> Self {
        return lgamma_r(self).0;
    }
}

