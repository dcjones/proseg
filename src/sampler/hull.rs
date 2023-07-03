
// Quickhull algorithm
// geo's implementation: https://docs.rs/geo/latest/src/geo/algorithm/convex_hull/qhull.rs.html#21-63

use std::cell::RefMut;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::DerefMut;


// All this is just a mechanism to to do static dispatch on whether we are above or
// below the line in quickhull_part.
trait QuickhullSide {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool;
}

#[derive(Debug)]
struct QuickhullAbove;

impl QuickhullSide for QuickhullAbove {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool {
        return !isabove(u, w, p) && !isabove(w, v, p)
    }
}

#[derive(Debug)]
struct QuickhullBelow;

impl QuickhullSide for QuickhullBelow {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool {
        return isabove(u, w, p) && isabove(w, v, p)
    }
}


/// Compute the convex hull and return it's area.
pub fn convex_hull_area(vertices: &mut RefMut<Vec<(f32,f32)>>, hull: &mut RefMut<Vec<(f32,f32)>>) -> f32 {
    if vertices.len() < 3 {
        hull.clear();
        hull.extend(vertices.iter().cloned());
        return 0.0;
    }

    // find the leftmost and rightmost points
    let (l, r) = horizontal_extrema_indices(&vertices);
    let (u, v) = (vertices[l], vertices[r]);

    hull.clear();
    hull.push(u);
    hull.push(v);

    // put l and r as the first two elements
    vertices.swap(0, l);
    if r == 0 {
        vertices.swap(1, l);
    } else {
        vertices.swap(1, r);
    }

    // partition into above and below the l,r line
    {
        let mut i = 1;
        let mut j = vertices.len();
        loop {
            i += 1;
            while (i < j) && isabove(u, v, vertices[i]) {
                i += 1;
            }

            j -= 1;
            while (j > i) && !isabove(u, v, vertices[j]) {
                j -= 1;
            }

            if i >= j {
                break;
            }

            vertices.swap(i, j);
        }

        quickhull_part(QuickhullAbove{}, &mut vertices[2..i], hull, u, v);
        quickhull_part(QuickhullBelow{}, &mut vertices[i..], hull, u, v);
    }

    // compute the area
    return polygon_area(hull.deref_mut());
}

pub fn polygon_area(vertices: &mut [(f32, f32)]) -> f32 {
    let c = center(&vertices);
    vertices.sort_unstable_by(|a, b| clockwise_cmp(c, *a, *b) );

    let mut area = 0.0;

    for (i, u) in vertices.iter().enumerate() {
        let j = (i + 1) % vertices.len();
        let v = vertices[j];

        // triangle formula.
        // area += u.0 * v.1 - v.0 * u.1;

        // trapezoid formula (this is more numerically stable with large coordinates)
        area += (v.0 + u.0) * (v.1 - u.1);
    }
    area = area.abs() / 2.0;

    return area;
}

fn center(vertices: &[(f32,f32)]) -> (f32, f32) {
    let mut x = 0.0;
    let mut y = 0.0;
    for v in vertices {
        x += v.0;
        y += v.1;
    }
    x /= vertices.len() as f32;
    y /= vertices.len() as f32;
    return (x, y);
}

fn clockwise_cmp(c: (f32, f32), a: (f32, f32), b: (f32, f32)) -> Ordering {
    // From: https://stackoverflow.com/a/6989383
    if a.0 - c.0 >= 0.0 && b.0 - c.0 < 0.0 {
        return Ordering::Less;
    } else if a.0 - c.0 < 0.0 && b.0 - c.0 >= 0.0 {
        return Ordering::Greater;
    } else if a.0 - c.0 == 0.0 && b.0 - c.0 == 0.0 {
        if a.1 - c.1 >= 0.0 || b.1 - c.1 >= 0.0 {
            return a.1.partial_cmp(&b.1).unwrap();
        } else {
            return b.1.partial_cmp(&a.1).unwrap();
        }
    }

    // compute the cross product of vectors (c -> a) x (c -> b)
    let det = (a.0 - c.0) * (b.1 - c.1) - (b.0 - c.0) * (a.1 - c.1);

    if det < 0.0 {
        return Ordering::Less;
    } else if det > 0.0 {
        return Ordering::Greater;
    } else {
        // points a and b are on the same line from the c
        // check which point is closer to the c
        let d1 = (a.0 - c.0) * (a.0 - c.0) + (a.1 - c.1) * (a.1 - c.1);
        let d2 = (b.0 - c.0) * (b.0 - c.0) + (b.1 - c.1) * (b.1 - c.1);
        if d1 >= d2 {
            return Ordering::Greater;
        } else {
            return Ordering::Less;
        }
    }
}


fn quickhull_part<T>(side: T, vertices: &mut [(f32, f32)], hull: &mut RefMut<Vec<(f32,f32)>>, u: (f32, f32), v: (f32, f32)) where T: QuickhullSide + Debug {
    if vertices.len() == 0 {
        return;
    }

    // find furthest point from the u,v line
    let mut k = 0;
    let mut maxdist = 0.0;
    for (i, w) in vertices.iter().enumerate() {
        let dist = linedist(u, v, *w);
        if dist > maxdist {
            k = i;
            maxdist = dist;
        }
    }

    let w = vertices[k];
    hull.push(vertices[k]);
    vertices.swap(0, k);

    let mut i = 0;
    let mut j = vertices.len();
    loop {
        i += 1;
        while (i < j) && T::tricontains(u, v, w, vertices[i]) {
            i += 1;
        }

        j -= 1;
        while (j > i) && !T::tricontains(u, v, w, vertices[j]) {
            j -= 1;
        }

        if i >= j {
            break;
        }

        vertices.swap(i, j);
    }

    quickhull_part(side, &mut vertices[i..], hull, u, v);

}


fn horizontal_extrema_indices(vertices: &[(f32,f32)]) -> (usize, usize) {
    let (mut i_min, mut i_max) = (0, 0);
    for (i, v) in vertices.iter().enumerate() {
        if v.0 < vertices[i_min].0 {
            i_min = i;
        }
        if v.0 > vertices[i_max].0 {
            i_max = i;
        }
    }

    return (i_min, i_max);
}

// Compute the distance from the line uv to the point w.
fn linedist(u: (f32, f32), v: (f32, f32), w: (f32, f32)) -> f32 {
    let a = v.0 - u.0;
    let b = v.1 - u.1;
    return (a * (u.1 - w.1) - b * (u.0 - w.0)).abs() / (a * a + b * b).sqrt()
}

// Test if the point w is above the line from u to v.
fn isabove(u: (f32, f32), v: (f32, f32), w: (f32, f32)) -> bool {
    return (v.0 - u.0) * (w.1 - u.1) - (v.1 - u.1) * (w.0 - u.0) > 0.0;
}
